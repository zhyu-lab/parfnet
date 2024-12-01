
import os
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch


class TensorboardWriter(SummaryWriter):
    def __init__(self,
                 exp,
                 PATH,
                 fig_path,
                 num_data=48,
                 clear=None):
        super().__init__()
        self.fig_path = fig_path
        self.num_data = num_data
        self.exp = exp

        if clear is not None:
            self.clear_Tensorboard(clear)

    def write_results(self,
                      keys: list,
                      results_train,
                      results_test,
                      epoch):
        for metric, index in zip(keys, range(len(results_test))):
            self.add_scalars(metric, {'Training': results_train[index],
                                      'Validation': results_test[index]},
                             epoch + 1)

    def write_images(self,
                    keys: list,
                    data: list,
                    step,
                    C=3,
                    best=True):

        rand_images = self.get_random_predictions(data=data,
                                                  num_data=self.num_data)


        image = rand_images[0]

        target = rand_images[1]
        prediction = rand_images[2]

        if C == 1:
            target_hot, pred_hot = target.squeeze(1), prediction.squeeze(1)
        elif C == 3:
            prediction = prediction.unsqueeze(1)
            target_hot = torch.eye(C)[target.type(
                torch.LongTensor).squeeze(1)].permute(0, 3, 1, 2)
            pred_hot = torch.eye(C)[prediction.type(
                torch.LongTensor).squeeze(1)].permute(0, 3, 1, 2)
        else:
            prediction = prediction.unsqueeze(1)
            target_hot, pred_hot = target, prediction
                        
        images = [image,
                    target_hot,
                    pred_hot
                    ]
        if best:
            self.visualize(data=images,
                           step=step)


    def visualize(self, data, step):
        plt.ioff()

        visualizations_path = self.fig_path + 'visualizations/'
        if not os.path.exists(visualizations_path):
            os.mkdir(visualizations_path)

        data_folder = visualizations_path + 'data/'
        prediction_folder = visualizations_path + 'prediction/'
        target_folder = visualizations_path + 'target/'

        for folder_path in [data_folder, prediction_folder, target_folder]:
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

        data_step_folder = data_folder + str(step) + '/'
        prediction_step_folder = prediction_folder + str(step) + '/'
        target_step_folder = target_folder + str(step) + '/'

        for step_folder_path in [data_step_folder, prediction_step_folder, target_step_folder]:
            if not os.path.exists(step_folder_path):
                os.mkdir(step_folder_path)


        for i, img_data in enumerate(data[0]):
            img_data = img_data.permute(1, 2, 0)
            if len(img_data.shape) != 3:
                img_data = img_data[:, :, 0]


            img_data_pil = Image.fromarray((img_data.cpu().numpy() * 255).astype('uint8'))
            img_data_pil = img_data_pil.resize((224, 224), Image.Resampling.LANCZOS)

            fig_data = plt.figure(figsize=(224 / 100, 224 / 100))
            plt.imshow(img_data_pil)
            file_name = data_step_folder + f'_data_{i}.png'
            fig_data.savefig(file_name)
            plt.close(fig_data)


        for i, img_target in enumerate(data[1]):

            img_target = img_target

            img_target_pil = Image.fromarray((img_target.cpu().numpy() * 255).astype('uint8'))
            img_target_pil = img_target_pil.resize((224, 224), Image.Resampling.LANCZOS)

            fig_tar = plt.figure(figsize=(224 / 100, 224 / 100))
            plt.imshow(img_target_pil, cmap='gray')
            file_name = target_step_folder + f'_target_{i}.png'
            fig_tar.savefig(file_name)
            plt.close(fig_tar)


        for i, img_pred in enumerate(data[2]):

            img_pred = img_pred

            img_pred_pil = Image.fromarray((img_pred.cpu().numpy() * 255).astype('uint8'))
            img_pred_pil = img_pred_pil.resize((224, 224), Image.Resampling.LANCZOS)

            fig_pred = plt.figure(figsize=(224 / 100, 224 / 100))
            plt.imshow(img_pred_pil, cmap='gray')
            file_name = prediction_step_folder + f'_prediction_{i}.png'
            fig_pred.savefig(file_name)
            plt.close(fig_pred)

    def write_hyperparams(self,
                          hparams_dict,
                          metric_dict):

        self.add_hparams(hparam_dict=hparams_dict,
                         metric_dict=metric_dict)

    def write_histogram(self):
        pass

    @staticmethod
    def clear_Tensorboard(file):
        dir = 'runs/' + file
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

    @staticmethod
    def get_random_predictions(data: list,
                               num_data=36):

        if data[0].shape[0] >= num_data:
            seed = torch.arange(num_data)
        else:
            seed = torch.arange(data[0].shape[0])            
        random_data = [i[seed] for i in data]
        return random_data
