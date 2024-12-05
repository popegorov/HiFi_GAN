from pathlib import Path

import pandas as pd
import itertools

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.loss.loss_utils import feature_loss, generator_loss, mel_spect_loss, discriminator_loss
from src.trainer.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.d_optimizer.zero_grad()

        source_wav = batch['audio'].unsqueeze(1)
        source_spec = batch['spectrogram']
        generated_wav = self.generator(source_spec)

        mpd_real_out, mpd_gen_out, _, _ = self.mpd(source_wav, generated_wav.detach())

        msd_real_out, msd_gen_out, _, _ = self.msd(source_wav, generated_wav.detach())

        mpd_loss_d = discriminator_loss(mpd_real_out, mpd_gen_out)
        msd_loss_d = discriminator_loss(msd_real_out, msd_gen_out)
        d_loss = mpd_loss_d + msd_loss_d

        if self.is_train:
            d_loss.backward()
            self._clip_grad_norm(self.mpd)
            self._clip_grad_norm(self.msd)
            self.d_optimizer.step()
            self.train_metrics.update("discriminator_grad_norm", self._get_grad_norm(itertools.chain(self.msd.parameters(), self.mpd.parameters())))
            self.g_optimizer.zero_grad()

        _, mpd_gen_out, mpd_real_features, mpd_gen_features = self.mpd(source_wav, generated_wav)
        _, msd_gen_out, msd_real_features, msd_gen_features = self.msd(source_wav, generated_wav)

        mpd_loss_g = generator_loss(mpd_gen_out)
        msd_loss_g = generator_loss(msd_gen_out)

        mel_spec_loss = mel_spect_loss(source_spec, self.mel_spec(generated_wav))

        mpd_loss_f = feature_loss(mpd_real_features, mpd_gen_features)
        msd_loss_f = feature_loss(msd_real_features, msd_gen_features)

        g_loss = mpd_loss_g + msd_loss_g + mel_spec_loss + mpd_loss_f + msd_loss_f
        
        if self.is_train:
            g_loss.backward()
            self._clip_grad_norm(self.generator)
            self.g_optimizer.step()
            self.train_metrics.update("generator_grad_norm", self._get_grad_norm(self.generator.parameters()))
            
        
        to_update = {
            "MPD_discriminator loss" : mpd_loss_d,
            "MSD_discriminator loss" : msd_loss_d,
            "MPD_generator loss" : mpd_loss_g,
            "MSD_generator loss" : msd_loss_g,
            "MPD_feature loss" : mpd_loss_f / 2.0, # returning to real value
            "MSD_feature loss" : msd_loss_f / 2.0, # returning to real value           
            "MelSpectrogram loss" : mel_spec_loss / 45.0, # returning to real value
            "Generator loss" : g_loss,
            "Discriminator loss" : d_loss,
            "generated_audio" : generated_wav.squeeze(1),
        }

        batch.update(to_update)

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        self.log_spectrogram(**batch)
        self.log_audio(**batch)

    def log_audio(self, audio, generated_audio, **batch):
        self.writer.add_audio("source_audio", audio[0], 22050)
        self.writer.add_audio("generated_audio", generated_audio[0], 22050)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        pass
        # TODO add beam search
        # Note: by improving text encoder and metrics design
        # this logging can also be improved significantly

        # argmax_inds = log_probs.cpu().argmax(-1).numpy()
        # argmax_inds = [
        #     inds[: int(ind_len)]
        #     for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        # ]
        # tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))

        # rows = {}
        # for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:

        #     wer = calc_wer(target, pred) * 100
        #     cer = calc_cer(target, pred) * 100

        #     rows[Path(audio_path).name] = {
        #         "target": target,
        #         "raw prediction": raw_pred,
        #         "predictions": pred,
        #         "wer": wer,
        #         "cer": cer,
        #     }
        # self.writer.add_table(
        #     "predictions", pd.DataFrame.from_dict(rows, orient="index")
        # )
