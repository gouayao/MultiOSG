from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            dataroot="./datasets/arches",
            dataset_mode="imagefolder",
            checkpoints_dir="./checkpoints/",
            num_gpus=1, batch_size=2,
            # scale the image such that the short side is |load_size|, and
            # crop a square window of |crop_size|.
            preprocess="scale_shortside_and_crop",
            model="multi",
            optimizer="swapping_autoencoder",
            resume_iter="latest",
            total_nimgs=100 * (100 ** 2) + 1,
            save_freq=5000,
            evaluation_freq=10000,
            load_size=512, crop_size=512,
            display_freq=1600, print_freq=480,
        )

        return [
            opt.specify(
                name="2_arches_default",
            ),
        ]

    def train_options(self):
        common_options = self.options()
        return [opt.specify(
            continue_train=True,
            evaluation_metrics="none") for opt in common_options]

    def test_options(self):
        opt = self.options()[0]
        return [
            # Swapping Grid Visualization. Fig 12 of the arxiv paper
            opt.tag("swapping_grid").specify(
                num_gpus=1,
                batch_size=1,
                dataroot="./testphotos/arches/",
                dataset_mode="imagefolder",
                preprocess="scale_shortside",  # For testing, scale but don't crop
                # preprocess="scale_width",  # For testing, scale but don't crop
                # load_size=223, crop_size=223,
                evaluation_metrics="structure_style_grid_generation"
            ),

            # Simple Swapping code for quick testing
            opt.tag("simple_swapping").specify(
                num_gpus=1,
                batch_size=1,
                dataroot=".",  # dataroot is ignored.
                evaluation_metrics="simple_swapping",
                # Specify the two images here.

                input_structure_image="./datasets/morhping/1.jpg",
                input_texture_image="./datasets/sigd16/7.jpg",
                # alpha == 1.0 corresponds to full swapping.
                # 0 < alpha < 1 means interpolation
                texture_mix_alpha=1.0,
            ),

            opt.tag("fake").specify(
                num_gpus=1,
                batch_size=1,
                dataroot="./testphotos/mou/",
                dataset_mode="imagefolder",
                preprocess="scale_shortside",  # For testing, scale but don't crop
                # load_size=256, crop_size=256,
                evaluation_metrics="fake"
            ),

            # Simple interpolation images for quick testing
            opt.tag("simple_interpolation").specify(
                num_gpus=1,
                batch_size=1,
                dataroot=".",  # dataroot is ignored.
                evaluation_metrics="simple_swapping",
                # Specify the two images here.
                input_structure_image="./testphotos/arches/structure/3.jpg",
                input_texture_image="./testphotos/arches/style/5.jpg",
                preprocess="scale_shortside",  # For testing, scale but don't crop
                load_size=256, crop_size=256,
                texture_mix_alpha='0.0 0.2 0.4 0.6 0.8 1.0',
            )
        ]
