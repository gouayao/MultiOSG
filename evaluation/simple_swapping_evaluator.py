import os
import torchvision.transforms as transforms
from PIL import Image
from evaluation import BaseEvaluator
from data.base_dataset import get_transform
import util
import torch
from models import BaseModel
import models.networks as networks
from models.single_model import SingleModel
import random
from torch.nn import functional as F
from torch import nn, autograd, optim


class SimpleSwappingEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--input_structure_image", required=True, type=str)
        parser.add_argument("--input_texture_image", required=True, type=str)
        parser.add_argument("--texture_mix_alphas", type=float, nargs='+',
                            default=[1.0],
                            help="Performs interpolation of the texture image."
                            "If set to 1.0, it performs full swapping."
                            "If set to 0.0, it performs direct reconstruction"
                            )
        
        opt, _ = parser.parse_known_args()
        dataroot = os.path.dirname(opt.input_structure_image)
        
        # dataroot and dataset_mode are ignored in SimpleSwapplingEvaluator.
        # Just set it to the directory that contains the input structure image.
        parser.set_defaults(dataroot=dataroot, dataset_mode="imagefolder")

        # parser.add_argument("--rate", default=0.5, type=float)
        
        return parser
    
    def load_image(self, path):
        path = os.path.expanduser(path)
        img = Image.open(path).convert('RGB')
        transform = get_transform(self.opt)
        tensor = transform(img).unsqueeze(0)
        return tensor

    def get_random(self, x, crop_window=None):
        """ Make random crops.
            Corresponds to the yellow and blue random crops of Figure 2.
        """
        crops = util.apply_random_crop(
            x, self.opt.patch_size*2,
            (self.opt.patch_min_scale * 8 * 0.6, self.opt.patch_max_scale * 4 * 0.6),
            num_crops=self.opt.patch_num_crops
        )
        return crops
    
    def evaluate(self, model, dataset, nsteps=None):
        structure_image = self.load_image(self.opt.input_structure_image)
        texture_image = self.load_image(self.opt.input_texture_image)
        os.makedirs(self.output_dir(), exist_ok=True)
        self.sim = nn.CosineSimilarity()
        self.sfm = nn.Softmax(dim=0)

        # for ii in range(0, 50):
        #     model(sample_image=structure_image, command="fix_noise")
        #     structure_code, source_texture_code = model(
        #         structure_image, command="encode")
        #     _, target_texture_code = model(texture_image, command="encode")
        #
        #     fake_sp = self.mix(structure_code, 8)
        #
        #     # # output_image, _ = model(mix1_sp, mix1_gl, command="decode")
        #     # output_image, _ = model(structure_code, source_texture_code, command="decode")
        #     # # print(',,,,,,,,,,,,,,,,,,,,,,,,',output_image.shape)
        #     # output_image = transforms.ToPILImage()(
        #     #     (output_image[0].clamp(-1.0, 1.0) + 1.0) * 0.5)
        #     # output_name = "%s%s.jpg" %( os.path.splitext(os.path.basename(self.opt.input_structure_image))[0], '_A')
        #     # output_path = os.path.join(self.output_dir(), output_name)
        #     # output_image.save(output_path)
        #     # print("Saved at " + output_path)
        #
        #
        #     # ######real
        #     # output_image = transforms.ToPILImage()(
        #     #     (structure_image[0].clamp(-1.0, 1.0) + 1.0) * 0.5)
        #     # output_name = "%s%s.jpg" %( os.path.splitext(os.path.basename(self.opt.input_structure_image))[0], '_real_' + str(ii))
        #     # # output_name = "%s.jpg" % (str(ii+1))
        #     # output_path = os.path.join(self.output_dir(), output_name)
        #     # output_image.save(output_path)
        #     # print("Saved at " + output_path)
        #
        #     ###########fake
        #     output_image, _ = model(fake_sp, source_texture_code, command="decode")
        #     output_image = transforms.ToPILImage()(
        #         (output_image[0].clamp(-1.0, 1.0) + 1.0) * 0.5)
        #     # output_image = output_image.resize((248,186))
        #     output_name = "%s%s.jpg" %( os.path.splitext(os.path.basename(self.opt.input_structure_image))[0], '_fake_' + str(ii))
        #     output_path = os.path.join(self.output_dir(), output_name)
        #     output_image.save(output_path)
        #     print("Saved at " + output_path)

        model(sample_image=structure_image, command="fix_noise")
        source_structure_code, source_texture_code = model(structure_image, command="encode")
        target_structure_code, target_texture_code = model(texture_image, command="encode")

        alphas = self.opt.texture_mix_alphas
        for alpha in alphas:
            texture_code = util.lerp(source_texture_code, target_texture_code, alpha)

            structure_code = util.lerp(source_structure_code,target_structure_code, alpha)

            # structure_code = torch.cat([source_structure_code,target_structure_code],dim=3)

            output_image, _ = model(structure_code, texture_code, command="decode")
            # output_image,_ = model(source_structure_code, texture_code, command="decode")
            # output_image, _ = model(structure_code, source_texture_code, command="decode")

            output_image = transforms.ToPILImage()(
                (output_image[0].clamp(-1.0, 1.0) + 1.0) * 0.5)

            output_name = "%s_%s_%.2f.jpg" % (
                os.path.splitext(os.path.basename(self.opt.input_structure_image))[0],
                os.path.splitext(os.path.basename(self.opt.input_texture_image))[0],
                alpha
            )

            output_path = os.path.join(self.output_dir(), output_name)

            output_image.save(output_path)
            print("Saved at " + output_path)

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

    def fake(self, feat, refs):

        # take refs:(N*8, 8, 16, 16) for example
        n, c, h, w = refs.size() #N*8, 8, 16, 16
        refs = refs.view(n, c, -1) # (N*8, 8, 256)

        nn, c, h, w = feat.size() # (N, 8, 16, 16)
        feat = feat.view(nn, c, -1)  # (N, 8, 256)
        fake_sp = []

        for v in range(nn):
            feat_v = feat[v:v+1, :, :]  # (1, 8, 256)
            fake_s = []
            for j in range(c):
                feat_s = feat_v[:, j:j + 1, :]  # (1, 1, 256)
                b = -1
                for k in range(int(n/nn)):
                    ref_s = refs[k + v*int(n/nn):k + v*int(n/nn)+ 1, j:j + 1, :]  # (1*1*256)
                    s = F.cosine_similarity(feat_s.squeeze(), ref_s.squeeze(), dim=0)
                    if b < s:
                        a = ref_s
                        b = s
                    else:
                        a = bb
                    bb = a
                fake_s.append(a)
            fake_s = torch.cat([item for item in fake_s], dim=1)  # (1*8*256)
            fake_s = fake_s.view(-1, c, h, w)  # (1, 8, 16, 16)
            fake_sp.append(fake_s)
        fake_sp = torch.cat([item for item in fake_sp], dim=0)  # (nn, 8, 16, 16)

        # lamb = torch.distributions.beta.Beta(alpha, alpha).sample()
        # lamb = torch.min(lamb, 1 - lamb)
        # fake_sp = lamb * feat.view(nn, c, h, w) + (1 - lamb) * fake_sp

        return fake_sp



