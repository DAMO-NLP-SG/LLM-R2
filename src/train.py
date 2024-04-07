import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import ast
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from torch import optim
from transformers.tokenization_utils_base import PaddingStrategy

from encoder import *
import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed, default_data_collator
)
from transformers.trainer_utils import is_main_process
from transformers.file_utils import cached_property, is_torch_available, is_torch_tpu_available
from models import QueryformerForCL
from trainers import CLTrainer

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# pre_lang_model = SentenceTransformer('all-MiniLM-L6-v2')


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    ####lz changed here, added the fixed pre-trained path
    # c_model_name_or_path: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "The model checkpoint for weights initialization."
    #                 "Don't set if you want to train a model from scratch."
    #     },
    # )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    ####lz change filter threshold here
    phi: float = field(
        default=0.95,
        metadata={
            "help": "Weight for instance weighting."
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The training data file (.txt or .csv)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    # todo: evaluation?
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    @cached_property
    # @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            self._n_gpu = 0
        # todo: uncomment local rank later
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print('filter chosen: ', model_args.phi)

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/",
                                delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")

    if model_args.model_name_or_path:
        # todo: further iters with pretrain, what about optimizers and other attributes?
        model = QueryformerForCL()
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        checkpoint = torch.load('simcse_result/' + model_args.model_name_or_path + '/pytorch_model.bin')
        # print(checkpoint.keys())
        model.load_state_dict(checkpoint)
        # optimizer.load_state_dict(checkpoint['opt'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        # model = QueryformerForCL.from_pretrained()
        # fix_bert = RobertaModel.from_pretrained(model_args.c_model_name_or_path)
        # model.fix_bert.load_state_dict(fix_bert.state_dict())
    else:
        # raise NotImplementedError
        logger.info("Training new model from scratch")
        model = QueryformerForCL()

    # model.resize_token_embeddings(len(tokenizer))

    # Prepare features
    column_names = datasets["train"].column_names
    sent2_cname = None
    if len(column_names) == 3:
        # Pair datasets
        db_id_cname = column_names[0]
        sent0_cname = column_names[1]
        sent1_cname = column_names[2]
    elif len(column_names) == 4:
        # Pair datasets with hard negatives
        db_id_cname = column_names[0]
        sent0_cname = column_names[1]
        sent1_cname = column_names[2]
        sent2_cname = column_names[3]
    elif len(column_names) == 2:
        # Unsupervised datasets
        db_id_cname = column_names[0]
        sent0_cname = column_names[1]
        sent1_cname = column_names[1]
    else:
        raise NotImplementedError

    def conv_dict(in_dict):
        # print(in_dict)
        for key in list(in_dict.keys()):
            # print(in_dict[key])
            in_dict[key] = ast.literal_eval(in_dict[key].replace('-inf', '-2e308'))
        return in_dict

    def prepare_features(examples):
        total = len(examples[sent0_cname])
        # Avoid "None" fields
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            else:
                examples[sent0_cname][idx] = conv_dict(ast.literal_eval(examples[sent0_cname][idx].replace('-inf', '-2e308')))
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "
            else:
                examples[sent1_cname][idx] = conv_dict(ast.literal_eval(examples[sent1_cname][idx].replace('-inf', '-2e308')))

        sentences = [examples[sent0_cname], examples[sent1_cname]]
        # db_ids = examples[db_id_cname]

        # If hard negative exists
        if sent2_cname is not None:
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = " "
                else:
                    examples[sent2_cname][idx] = conv_dict(ast.literal_eval(examples[sent2_cname][idx].replace('-inf', '-2e308')))
            sentences.append(examples[sent2_cname])
        sentences = np.array(sentences).T.tolist()
        # sent_features = prepare_enc_data(sentences, pre_lang_model, db_ids)
        sent_features = collator(sentences)
        # print(sent_features)
        return sent_features

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
            # load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    @dataclass
    class OurDataCollatorWithPadding:
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[
            str, torch.Tensor]:
            # print(len(features[0]['x']))
            # print(features[0]['x'][0].size())

            x_row = []
            attn_bias_row = []
            rel_pos_row = []
            heights_row = []
            for row in features:
                x_row.append(torch.FloatTensor(row['x']))
                attn_bias_row.append(torch.FloatTensor(row['attn_bias']))
                rel_pos_row.append(torch.LongTensor(row['rel_pos']))
                heights_row.append(torch.LongTensor(row['heights']))
            x = torch.squeeze(torch.cat(x_row))
            attn_bias = torch.squeeze(torch.cat(attn_bias_row))
            rel_pos = torch.squeeze(torch.cat(rel_pos_row))
            heights = torch.squeeze(torch.cat(heights_row))
            out_features = {'attn_bias': attn_bias, 'rel_pos': rel_pos, 'heights': heights, 'x': x}
            return out_features

    data_collator = OurDataCollatorWithPadding()

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        # tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.model_args = model_args

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

        # Evaluation
        results = {}
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            results = trainer.evaluate(eval_senteval_transfer=True)

            output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key, value in sorted(results.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
