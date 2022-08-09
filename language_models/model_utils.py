from transformers import ElectraConfig
from transformers import AutoModelForMaskedLM, BertConfig, BertForMaskedLM
from transformers import Trainer, TrainingArguments, TrainerCallback

from language_models.evaluation_utils import eval_rocauc
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from torch.optim import AdamW

import torch
import torch.nn as nn
import torch.nn.functional as F

eval_epochs = [5, 10]


def configure_model(architecture, pretrained, small, vocab_size, tokenizer, embed_size, size=1):
    """
    Configure a BERT model for a given configuration
    """
    if architecture == "bert":
        if pretrained:
            model_checkpoint = "bert-base-uncased"
            config = BertConfig.from_pretrained(model_checkpoint)
        else:
            if small:
                config = BertConfig(
                    pad_token_id=tokenizer.pad_token_id,
                    num_hidden_layers=2 * size,
                    intermediate_size=192 * size,
                    num_attention_heads=6 * size,
                    hidden_size=120 * size,
                    max_position_embeddings=embed_size,
                    # hidden_dropout_prob=0.3,
                    # attention_probs_dropout_prob=0.3,
                    # classifier_dropout=0.3,
                    # initializer_range=0.02,
                    vocab_size=vocab_size,
                )
            else:
                config = BertConfig(vocab_size=vocab_size)
        model = BertForMaskedLM(config)

    elif architecture == "electra":
        model_checkpoint = "electra-base-generator"

        if pretrained:
            config = ElectraConfig.from_pretrained(
                "google/electra-base-generator")
            config.is_decoder = True
            model = AutoModelForMaskedLM.from_pretrained(
                "google/electra-base-generator", config=config
            )
        else:
            config = ElectraConfig()
            config.is_decoder = True
            model = AutoModelForMaskedLM.from_config(config)

    model.init_weights()
    return model.cuda()


class EvalCallback(TrainerCallback):
    "Callback that evaluates ROCAUC score after every epoch"

    def __init__(
        self,
        save_model_path,
        train_set_name,
        tokenizer,
        model,
        dss_test,
        batch_size_eval,
        tb_writer,
        ds_name
    ):
        self.best_loss = float("inf")
        self.save_model_path = save_model_path
        self.train_set_name = train_set_name
        self.tokenizer = tokenizer
        self.model = model
        self.dss_test = dss_test
        self.batch_size_eval = batch_size_eval
        self.tb_writer = tb_writer
        self.ds_name = ds_name

    def on_evaluate(self, args, state, control, **kwargs):
        epoch = state.epoch
        loss = state.log_history[-1]["eval_loss"]
        if loss < self.best_loss:
            self.best_loss = loss
            print(f"Best loss at epoch {epoch}: {self.best_loss}")

        self.model.save_pretrained(self.save_model_path + f'_{epoch}')
        self.model.save_pretrained(self.save_model_path + "_final")

        self.model.eval()
        with torch.no_grad():
            if int(epoch) in eval_epochs:
                eval_rocauc(
                    model=self.model,
                    dss_test=self.dss_test,
                    bs_eval=self.batch_size_eval,
                    tokenizer=self.tokenizer,
                    epoch=epoch,
                    tb_writer=self.tb_writer,
                )


def distil_model(
    teacher,
    student,
    tokenizer,
    ds_train,
    dss_test,
    save_model_path,
    batch_size_train,
    batch_size_eval,
    num_epochs,
    tb_writer,
    temperature=4,
    alpha=0.9
):
    final_model_path = save_model_path + "_final"
    data_collator = DataCollatorForLanguageModeling(tokenizer)

    lm_dataloader_train = DataLoader(
        dataset=ds_train,
        batch_size=batch_size_train,
        sampler=RandomSampler(data_source=ds_train),
        num_workers=10,
    )

    KL = nn.KLDivLoss(reduction='batchmean')
    optim = AdamW(student.parameters(), lr=1e-5, weight_decay=0.01)

    for epoch in range(1, num_epochs+1):
        print("Starting epoch", epoch)
        epoch_loss_kl = 0.
        epoch_loss_ce = 0.
        epoch_loss = 0.

        for batch in tqdm(lm_dataloader_train):
            batch_cpy = {key: value for key, value in batch.items() if key not in [
                'idx', 'count', 'service']}
            batch_text = [" ".join(v) for v in list(zip(*batch_cpy.values()))]
            tokenized_text = tokenizer(batch_text)
            collated = data_collator(tokenized_text['input_ids'])
            input_ids = collated['input_ids']
            labels = collated['labels']

            outputs_s = student(input_ids=input_ids.cuda(),
                                labels=labels.cuda())
            logits_s = outputs_s.logits
            loss_s = outputs_s.loss

            with torch.no_grad():
                logits_t = teacher(input_ids=input_ids.cuda()).logits

            loss_kl = KL(input=F.log_softmax(logits_s/temperature, dim=-1),
                         target=F.softmax(logits_t/temperature, dim=-1))

            loss = (1 - alpha) * loss_s + alpha * loss_kl
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss_kl += loss_kl.item()
            epoch_loss_ce += loss_s.item()
            epoch_loss += loss.item()

        student.save_pretrained(save_model_path + f"_{epoch}.0")
        student.save_pretrained(final_model_path)

        ds_len = len(lm_dataloader_train)
        epoch_loss_kl /= ds_len
        epoch_loss_ce /= ds_len
        epoch_loss /= ds_len

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss_kl", epoch_loss_kl, epoch)
            tb_writer.add_scalar("train/loss_ce", epoch_loss_ce, epoch)
            tb_writer.add_scalar("train/loss", epoch_loss, epoch)

        if int(epoch) in eval_epochs:
            eval_rocauc(
                model=student,
                dss_test=dss_test,
                bs_eval=batch_size_eval,
                tokenizer=tokenizer,
                epoch=epoch,
                tb_writer=tb_writer,
            )

    return student


def train_model(
    model,
    tokenizer,
    ds_name,
    train_set_name,
    run_name,
    lm_ds_train,
    lm_ds_eval,
    dss_test,
    save_model_path,
    batch_size_train,
    batch_size_eval,
    num_epochs,
    tb_writer,
):

    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=True, mlm_probability=0.15)
    training_args = TrainingArguments(
        save_model_path,
        per_device_train_batch_size=batch_size_train,
        per_device_eval_batch_size=batch_size_eval,
        do_eval=False,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        dataloader_num_workers=12,
        learning_rate=1e-5,
        lr_scheduler_type='constant',
        weight_decay=0.01,
        seed=42,
        num_train_epochs=num_epochs,
        metric_for_best_model="loss",
        save_strategy="no",  # we handle saving
        logging_dir="runs/" + run_name,
        optim='adamw_torch',
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=lm_ds_train,
        eval_dataset=lm_ds_eval,
        callbacks=[
            EvalCallback(
                save_model_path=save_model_path,
                train_set_name=train_set_name,
                tokenizer=tokenizer,
                model=model,
                dss_test=dss_test,
                batch_size_eval=batch_size_eval,
                tb_writer=tb_writer,
                ds_name=ds_name
            ),
        ],
    )

    print("Training started...")
    trainer.train()
