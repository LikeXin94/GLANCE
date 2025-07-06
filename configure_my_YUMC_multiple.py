def get_default_config(data_name):
    if data_name in ['digit_6view']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                arch3=[128, 256, 128],
                arch4=[128, 256, 128],
                arch5=[128, 256, 128],
                arch6=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[76, 1024, 1024, 1024, 128],
                arch2=[216, 1024, 1024, 1024, 128],
                arch3=[64, 1024, 1024, 1024, 128],
                arch4=[240, 1024, 1024, 1024, 128],
                arch5=[47, 1024, 1024, 1024, 128],
                arch6=[6, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',
                activations5='relu',
                activations6='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                batch_size=256,
                epoch=100,
                lr=1e-4,
                lambda1=[1],
                lambda2=[1e3],
                lambda3=[1e-3],
            ),
        )

    else:
        raise Exception('Undefined data_name')
