#!/usr/bin/env python3

import os
import pickle
import sys
from dataclasses import asdict
from ipaddress import ip_address
from pathlib import Path

import torch
import transformers
import requests
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch_optimizer import Lamb
from transformers import DataCollatorForLanguageModeling, HfArgumentParser, TrainingArguments, set_seed
from transformers.models.albert import AlbertConfig, AlbertForPreTraining, AlbertTokenizerFast
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer import Trainer
from transformers.trainer_utils import is_main_process

from hivemind import DHT, Float16Compression, Optimizer, get_dht_time
from hivemind.optim.state_averager import LRSchedulerBase
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from hivemind.utils.networking import log_visible_maddrs

import threading
import time
import utils
from arguments import (
    AlbertTrainingArguments,
    AveragerArguments,
    CollaborationArguments,
    DatasetArguments,
    ProgressTrackerArguments,
)

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


def setup_transformers_logging(process_rank: int):
    if is_main_process(process_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.disable_default_handler()
        transformers.utils.logging.enable_propagation()


def get_model(training_args, config, tokenizer):
    # Find latest checkpoint in output_dir
    output_dir = Path(training_args.output_dir)
    logger.info(f'Checkpoint dir {output_dir}, contents {list(output_dir.glob("checkpoint*"))}')
    latest_checkpoint_dir = max(output_dir.glob("checkpoint*"), default=None, key=os.path.getctime)

    if latest_checkpoint_dir is not None:
        logger.info(f"Loading model from {latest_checkpoint_dir}")
        model = AlbertForPreTraining.from_pretrained(latest_checkpoint_dir)
    else:
        logger.info(f"Training from scratch")
        model = AlbertForPreTraining(config)
        model.resize_token_embeddings(len(tokenizer))

    return model

def get_opt_and_scheduler(training_args, model):
    opt = lambda params: Lamb(
        params,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        clamp_value=training_args.clamp_value,
        debias=True,
    )

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    scheduler = lambda opt: get_linear_schedule_with_warmup(
        opt, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.total_steps
    )

    return opt, scheduler, params

def assist_averaging_in_background(
        lock: threading.Lock, optimizer: Optimizer, opt_args: CollaborationArguments, finished: threading.Event
):
    logger.info("Entered background assist in averaging")
    while not finished.is_set():
        try:
            time.sleep(opt_args.assist_refresh)
            optimizer.step()
        except Exception as e:
            logger.exception(e, exc_info=True)

def main():
    parser = HfArgumentParser(
        (
            AlbertTrainingArguments,
            DatasetArguments,
            CollaborationArguments,
            AveragerArguments,
            ProgressTrackerArguments,
        )
    )
    training_args, dataset_args, collaboration_args, averager_args, tracker_args = parser.parse_args_into_dataclasses()
    logger.info(f"Found {len(collaboration_args.initial_peers)} initial peers: {collaboration_args.initial_peers}")

    if collaboration_args.use_google_dns:
        request = requests.get("https://api.ipify.org")
        request.raise_for_status()

        address = request.text
        logger.info(f"Received public IP address of this machine: {address}")
        version = ip_address(address).version
        port = collaboration_args.host_maddrs[0].split('/')[-1]
        collaboration_args.announce_maddrs += [f"/ip{version}/{address}/tcp/{port}"]

    setup_transformers_logging(training_args.local_rank)
    logger.info(f"Training/evaluation parameters:\n{training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AlbertConfig.from_pretrained(dataset_args.config_path, cache_dir=dataset_args.cache_dir)
    try:
        tokenizer = AlbertTokenizerFast.from_pretrained(dataset_args.tokenizer_path, cache_dir=dataset_args.cache_dir)
    except OSError:
        logger.fatal(
            f"No tokenizer data found in {dataset_args.tokenizer_path}, "
            f"please run ./tokenize_wikitext103.py before running this"
        )
        sys.exit(1)

    model = get_model(training_args, config, tokenizer)
    model.to(training_args.device)

    opt, scheduler, params = get_opt_and_scheduler(training_args, model)

    #tokenized_datasets = load_from_disk(Path(dataset_args.dataset_path))
    # This data collator will take care of randomly masking the tokens.
    #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    validators, local_public_key = utils.make_validators(collaboration_args.run_id)

    dht = DHT(
        start=True,
        initial_peers=collaboration_args.initial_peers,
        client_mode=collaboration_args.client_mode,
        record_validators=validators,
        use_auto_relay=collaboration_args.use_auto_relay,
        use_relay=collaboration_args.use_relay,
        use_ipfs=collaboration_args.use_ipfs,
        host_maddrs=collaboration_args.host_maddrs,
        announce_maddrs=collaboration_args.announce_maddrs,
        identity_path=collaboration_args.identity_path,
    )
    log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=collaboration_args.use_ipfs)

    total_batch_size_per_step = None # aux peer does as little as possible
    adjusted_target_batch_size = collaboration_args.target_batch_size - collaboration_args.batch_size_lead

    optimizer = Optimizer(
        dht=dht,
        run_id=collaboration_args.run_id,
        target_batch_size=adjusted_target_batch_size,
        batch_size_per_step=total_batch_size_per_step,
        optimizer=opt,
        params=params,
        scheduler=scheduler,
        matchmaking_time=collaboration_args.matchmaking_time,
        averaging_timeout=collaboration_args.averaging_timeout,
        offload_optimizer=True,
        delay_optimizer_step=True,
        delay_grad_averaging=True,
        client_mode=collaboration_args.client_mode,
        grad_compression=Float16Compression(),
        state_averaging_compression=Float16Compression(),
        averager_opts={"bandwidth": collaboration_args.bandwidth, **asdict(averager_args)},
        tracker_opts=asdict(tracker_args),
        verbose=True,
        auxiliary=True
    )
    finished, lock = threading.Event(), threading.Lock()
    assert not collaboration_args.client_mode, "client-mode peers cannot assist in averaging"
    averaging_thread = threading.Thread(
        name="AveragingAuxThread", target=assist_averaging_in_background,
        args=[lock, optimizer, collaboration_args, finished], daemon=True
    )
    averaging_thread.start()

    run_id = collaboration_args.run_id
    current_step = 0

    while True:
        metrics_dict = dht.get(run_id + "_metrics", latest=True)
        if metrics_dict is not None:
            metrics_dict = metrics_dict.value
            metrics = [utils.LocalMetrics.parse_obj(metrics_dict[peer].value) for peer in metrics_dict]
            latest_step = max(item.step for item in metrics)

            if latest_step != current_step:
                logger.debug(f"Got metrics from {len(metrics)} peers")

                for i, metrics_for_peer in enumerate(metrics):
                    logger.debug(f"{i} peer {metrics_for_peer}")

                current_step = latest_step
                alive_peers = 0
                sum_loss = 0
                num_samples = 0
                sum_perf = 0
                sum_mini_steps = 0

                for item in metrics:
                    sum_loss += item.loss
                    alive_peers += 1
                    sum_perf += item.samples_per_second
                    num_samples += item.samples_accumulated
                    sum_mini_steps += item.mini_steps
                current_loss = sum_loss / sum_mini_steps
                logger.info(f"Step #{current_step}\tloss = {current_loss:.5f}")

        logger.debug("Peer is still alive...")
        time.sleep(collaboration_args.refresh_period)



if __name__ == "__main__":
    main()
