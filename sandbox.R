library(keras)
library(pROC)
library(caret)
library(tidyverse)
library(data.table)

train_dir      <- 'data/chest_xray/train/'
validation_dir <- 'data/chest_xray/val/'
test_dir       <- 'data/chest_xray/test/'



# Keras Generators --------------------------------------------------------

training_batch_size   <- 16
validation_batch_size <- 16

# Data Augmentation
train_datagen <-
	image_data_generator(
		rescale = 1/255,
		rotation_range = 25,
		width_shift_range = 0.1,
		height_shift_range = 0.05,
		shear_range = 0.15,
		zoom_range = 0.35,
		horizontal_flip = TRUE,
		vertical_flip = TRUE,
		fill_mode = "reflect"
	)

validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen       <- image_data_generator(rescale = 1/255)

train_generator <-
	flow_images_from_directory(
		train_dir,                            # Target directory
		train_datagen,                        # Data generator
		classes = c('NORMAL', 'PNEUMONIA'),
		target_size = c(224, 224),            # Resizes all images
		batch_size = training_batch_size,
		class_mode = "categorical",
		shuffle = T
	)

validation_generator <-
	flow_images_from_directory(
		validation_dir,
		classes = c('NORMAL', 'PNEUMONIA'),
		validation_datagen,
		target_size = c(224, 224),
		batch_size = validation_batch_size,
		class_mode = "categorical",
		shuffle = T
	)

test_generator <-
	flow_images_from_directory(
		test_dir,
		classes = c('NORMAL', 'PNEUMONIA'),
		test_datagen,
		target_size = c(224, 224),
		batch_size = 1,
		class_mode = "categorical",
		shuffle = FALSE
	)


# Transfer Learning -------------------------------------------------------

# Load base CNN
# conv_base <-
# 	application_inception_resnet_v2(
# 		weights = "data/keras-pretrained-models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5",
# 		include_top = FALSE,
# 		input_shape = c(224, 224, 3)
# )
#
conv_base <- application_vgg16(
	weights = "imagenet",
	include_top = FALSE,
	input_shape = c(224, 224, 3)
)

# We make sure the base is not trained again
freeze_weights(conv_base)

# Create new top
model <-
	keras_model_sequential() %>%
	conv_base %>%
	layer_flatten() %>%
	layer_dense(units = 224, activation = "relu", trainable = T) %>%
	layer_dropout(rate = 0.3) %>%
	layer_dense(units = 2, activation = "softmax", trainable = T)

model %>%
	compile(
		loss = "categorical_crossentropy",
		optimizer = optimizer_sgd(lr=1e-6),
		metrics = c("accuracy")
	)

training_step_size <- ceiling(length(list.files(train_dir, recursive = T)) / training_batch_size)
validation_step_size <- ceiling(length(list.files(validation_dir, recursive = T)) / validation_batch_size)


# Train -------------------------------------------------------------------

history <-
	model %>%
		fit_generator(
			train_generator,
			steps_per_epoch = 10,
			# class_weight = list("0"=1,"1"=pneumonia_adjustment),
			epochs = 10,
			validation_data = validation_generator,
			validation_steps = validation_step_size
		)


plot(history)


# Evaluate ----------------------------------------------------------------
#Make predictions on the test set
preds = predict_generator(model,
													test_generator,
													steps = length(list.files(test_dir, recursive = T)))

# Do some tidying
predictions = data.frame(test_generator$filenames)
predictions$prob_pneumonia = preds[,2]
colnames(predictions) = c('Filename', 'Prob_Pneumonia')

head(predictions, 10)

predictions$Class_predicted = 'Normal'
predictions$Class_predicted[predictions$Prob_Pneumonia >= 0.5] = 'Pneumonia'
predictions$Class_actual = 'Normal'
predictions$Class_actual[grep("PNEUMONIA", predictions$Filename)] = 'Pneumonia'

predictions$Class_actual = as.factor(predictions$Class_actual)
predictions$Class_predicted = as.factor(predictions$Class_predicted)

roc = roc(response = predictions$Class_actual,
					predictor = as.vector(predictions$Prob_Pneumonia),
					ci=T,
					levels = c('Normal', 'Pneumonia'))
threshold = coords(roc, x = 'best', best.method='youden')
threshold

#Boxplot
boxplot(predictions$Prob_Pneumonia ~ predictions$Class_actual,
				main = 'Probabilities of Normal vs Pneumoia',
				ylab = 'Probability',
				col = 'light blue')
abline(h=threshold[1], col = 'red', lwd = 3)

predictions$Class_predicted = 'Normal'
predictions$Class_predicted[predictions$Prob_Pneumonia >= threshold[1]] = 'Pneumonia'

#Create a confusion matrix
cm = confusionMatrix(predictions$Class_predicted %>% as.factor(), predictions$Class_actual, positive = 'Pneumonia')
cm

paste("AUC =", round(roc$auc, 3))
plot(roc)
ci_sens = ci.se(roc, specificities = seq(from=0, to=1, by=0.01), boot.n = 2000)
plot(ci_sens, type="shape", col="#00860022", no.roc=TRUE)
abline(h=cm$byClass['Sensitivity'], col = 'red', lwd = 2)
abline(v=cm$byClass['Specificity'], col = 'red', lwd = 2)

#Find high sensitivity points, but not so high that specificity is terrible,
thresholds = coords(roc, x = "local maximas")
thresholds = as.data.frame(t(thresholds))
thresholds[(thresholds$sensitivity > 0.85 & thresholds$sensitivity < 0.9),]
