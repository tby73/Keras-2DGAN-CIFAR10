import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

# random latent vector generator
def GenerateLatentPoints(latent_dim, n_samples):
    input_data = np.random.randn(latent_dim * n_samples)
    input_data = input_data.reshape(n_samples, latent_dim)
    return input_data

def ShowGenerations(generations, samples):
    plt.figure(figsize=(20, 20))

    for i in range(samples):
        # display images
        plt.subplot(samples, samples, i + 1)
        plt.imshow(generations[i, :, :, :])
    
    plt.show()

def ProcessGeneration(prediction):
    output = (prediction + 1) / 2.0
    output = (output * 255).astype(np.uint8)
    return output

def main():
    # load trained GAN model
    cifar_gan = keras.models.load_model("cifar_gan_v1.h5")

    # generate random latent vector
    random_latent_vector = GenerateLatentPoints(100, 25)

    # get and process predictions
    predictions = cifar_gan.predict(random_latent_vector)
    generated_images = ProcessGeneration(predictions)

    # Show 3 generated images
    ShowGenerations(generated_images, 3)

if __name__ == "__main__":
    main()

