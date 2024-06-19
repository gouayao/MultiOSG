from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            dataroot="./datasets/bc",
            dataset_mode="imagefolder",
            checkpoints_dir="./checkpoints/",
            num_gpus=1, batch_size=1,
            # scale the image such that the short side is |load_size|, and
            # crop a square window of |crop_size|.
            preprocess="scale_shortside_and_crop",
            model="one",
            optimizer="swapping_autoencoder",
            resume_iter="latest",
            total_nimgs=100 *(100 ** 2)+1,
            save_freq=5000,
            evaluation_freq=10000,
            load_size=256, crop_size=256,
            display_freq=1600, print_freq=480,
        )

        return [
            opt.specify(
                name="1_bc_default",
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
                dataroot="./testphotos/mix/",
                dataset_mode="imagefolder",
                evaluation_metrics="structure_style_grid_generation"
            ),
            
            # Simple Swapping code for quick testing
            opt.tag("simple_swapping").specify(
                num_gpus=1,
                batch_size=1,
                dataroot=".",  # dataroot is ignored.
                evaluation_metrics="simple_swapping",
                # Specify the two images here.

                input_structure_image="./datasets/SIGD/colusseum.png",
                input_texture_image="./datasets/SIGD/balloons.png",
                # alpha == 1.0 corresponds to full swapping.
                # 0 < alpha < 1 means interpolation
                texture_mix_alpha=1.0,
            ),

            opt.tag("fake").specify(
                num_gpus=1,
                batch_size=1,
                dataroot="./testphotos/balloons/",
                dataset_mode="imagefolder",
                preprocess="scale_shortside",  # For testing, scale but don't crop
                # preprocess="scale_width",  # For testing, scale but don't crop
                evaluation_metrics="fake"
            ),
            
            # Simple interpolation images for quick testing
            opt.tag("simple_interpolation").specify(
                num_gpus=1,
                batch_size=1,
                dataroot=".",  # dataroot is ignored.
                evaluation_metrics="simple_swapping",
                # Specify the two images here.
                input_structure_image="./datasets/SIGD/colusseum.png",
                input_texture_image="./datasets/SIGD/balloons.png",
                preprocess="scale_shortside",  # For testing, scale but don't crop
                texture_mix_alpha='0.0 0.25 0.5 0.75 1.0',
            )
        ]
