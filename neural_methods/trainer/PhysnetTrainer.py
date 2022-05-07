#TODO: Docstring
"""A one line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

  Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
from neural_methods.trainer.BaseTrainer import BaseTrainer
import torch
from torch.autograd import Variable
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm


class PhysnetTrainer(BaseTrainer):

    def __init__(self, config):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        logging.debug(self.device)
        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]
        self.loss_model = Neg_Pearson()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.TRAIN.LR)
        self.epochs = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        logging.debug(self.device)

    def train(self, data_loader):
        """ TODO:Docstring"""
        min_valid_loss = 1
        for round in range(self.epochs):
            logging.debug(f"====training:ROUND{round}====")
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)

            for i, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    Variable(batch[0]).to(torch.float32).to(self.device))
                BVP_label = Variable(batch[1]).to(
                    torch.float32).to(self.device)
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                    torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                loss_ecg.backward()
                train_loss.append(loss_ecg.item())
                self.optimizer.step()
                self.optimizer.zero_grad()
                vbar.set_postfix(loss=loss_ecg.item())
            train_loss = np.asarray(train_loss)
            logging.debug(np.mean(train_loss))

            valid_loss = self.valid(data_loader)
            # saves the model according to the loss on valid sets.
            if(valid_loss < min_valid_loss):
                min_valid_loss = valid_loss
                logging.debug("update best model")
                self.save_model()

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)

            for valid_i, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                #vbar.set_description("Valid %s" % valid_i)
                BVP_label = Variable(valid_batch[1]).to(
                    torch.float32).to(self.device)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    Variable(valid_batch[0]).to(torch.float32).to(self.device))
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                    torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
                train_loss.append(loss.item())
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss.item())
            print(np.mean(valid_loss))
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        logging.debug(" ====testing===")
        test_step = 0
        test_loss = []
        self.model.eval()
        with torch.no_grad():
            for test_i, test_batch in enumerate(data_loader["test"]):
                BVP_label = Variable(test_batch[1]).to(
                    torch.float32).to(self.device)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    Variable(test_batch[0]).to(torch.float32).to(self.device))
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                    torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                test_step += 1
                logging.debug(loss_ecg.item())
                test_loss.append(test_loss)
        return np.mean(test_loss)

    def save_model(self):
        if(not os.path.exists(self.model_dir)):
            os.makedirs(self.model_dir)
        torch.save(self.model.state_dict(), os.path.join(
            self.model_dir, self.model_file_name))

    # def load_model(self):
