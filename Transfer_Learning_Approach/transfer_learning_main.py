from keras.applications import vgg16

pretrained_model = vgg16.VGG16(include_top  = True,
	                           weights      = "imagenet",
	                           input_shape  = (20, 63, 1),
	                           classes      = 2)

print(pretrained_model.summary()) 