import os
import torch
import os
import torchvision.transforms as transforms
from evaluation import BaseEvaluator
from data.base_dataset import get_transform
import util
import numpy as np
from PIL import Image
from torch.nn import functional as F
from torch import nn, autograd, optim


class FakeEvaluator(BaseEvaluator):
    """ generate swapping images and save to disk """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def evaluate(self, model, dataset, nsteps=None):

        structure_images = {}
        for i, data_i in enumerate(dataset):
            bs = data_i["real_A"].size(0)
            for j in range(bs):
                image = data_i["real_A"][j:j + 1]
                path = data_i["path_A"][j]
                imagename = os.path.splitext(os.path.basename(path))[0]
                if "/structure/" in path:
                    structure_images[imagename] = image

        sps = []
        structure_paths = list(structure_images.keys())
        for structure_path in structure_paths:
            structure_image = structure_images[structure_path].cuda()
            sps.append(model(structure_image, command="encode")[0])


        for i, structure_path in enumerate(structure_paths):
            structure_image = structure_images[structure_path]

            for j in range(0, 10):
                model(sample_image=structure_image, command="fix_noise")
                structure_code, source_texture_code = model(
                    structure_image, command="encode")

                # ######## fake
                fake_sp = self.mix(structure_code, 8)
                # fake_sp2 = self.mix(structure_code, 8)
                # fake_sp = torch.cat([fake_sp1,structure_code,fake_sp2], dim=3)
                output_image, _ = model(fake_sp, source_texture_code, command="decode")
                output_image = transforms.ToPILImage()(
                    (output_image[0].clamp(-1.0, 1.0) + 1.0) * 0.5)
                # output_image = output_image.resize((248,186))


                ###real
                # output_image = transforms.ToPILImage()(
                #     (structure_image[0].clamp(-1.0, 1.0) + 1.0) * 0.5)



                # output_name = "%s.jpg" % (os.path.splitext(os.path.basename(structure_path))[0])
                # output_path = os.path.join(self.output_dir(), output_name)
                # output_image.save(output_path)
                # print("Saved at " + output_path)

                output_name = "%s" % (os.path.splitext(os.path.basename(structure_path))[0])
                output_name = "%s%s.jpg" % (output_name, "_" + str(j))
                output_path = os.path.join(self.output_dir(),str(i+1))
                os.makedirs(output_path, exist_ok=True)
                output_image.save(output_path + f'/{output_name}')
                print("Saved at " + output_path)

                # output_name = "%s.jpg" % (str(i * 10 + j + 1))
                # output_path = os.path.join(self.output_dir())
                # os.makedirs(output_path, exist_ok=True)
                # output_image.save(output_path + f'/{output_name}')
                # print("Saved at " + output_path)






            # ####real
            # model(sample_image=structure_image, command="fix_noise")
            # structure_code, source_texture_code = model(
            #     structure_image, command="encode")
            # fake_sp = self.mix(structure_code, 8)
            # output_image = transforms.ToPILImage()(
            #     (structure_image[0].clamp(-1.0, 1.0) + 1.0) * 0.5)
            # output_name = "%s.jpg" %( os.path.splitext(os.path.basename(structure_path))[0])
            # output_path = os.path.join(self.output_dir(), output_name)
            # output_image.save(output_path)
            # print("Saved at " + output_path)
        return {}

    def mix(self, feat, alpha):
        nn, c, h, w = feat.size()  # (N, 8, 16, 16)
        # feat = feat.view(nn, c, -1)  # (N, 8, 256)
        fake_sp = []
        for v in range(nn):
            feat_v = feat[v]  # (8, 16, 16)
            # feat_refs = feat_v.view(c, -1)  # (8, 256)
            fake_s = []
            for q in range(w // alpha):
                feat_q = feat_v[:, :, q * alpha:(q + 1) * alpha]  # (8, 16, 2)
                refs = feat_v[:, :, torch.randperm(feat_v.size()[2])]
                # refs = refs.view(c, h, w)  # (8, 16, 16)
                b = -1
                for k in range(w // alpha):
                    ref_s = refs[:, :, k * alpha:(k + 1) * alpha]  # (8, 16, 2)
                    s = F.cosine_similarity(feat_q.reshape(-1), ref_s.reshape(-1), dim=0)
                    if b < s:
                        a = ref_s
                        b = s
                    else:
                        a = bb
                    bb = a
                fake_s.append(a)
            fake_s = torch.cat([item for item in fake_s], dim=2)  # (8, 16, 16)
            fake_s = fake_s.unsqueeze(dim=0)  # (1, 8, 16, 16)
            fake_sp.append(fake_s)
        fake_sp = torch.cat([item for item in fake_sp], dim=0)  # (nn, 8, 16, 16)
        return fake_sp



