# Face Detection Using Python and Tensorflow 

- We are using the albumentations library and Using the Label_path to get the label in the model

                                    augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
                                                          alb.HorizontalFlip(p=0.5), 
                                                          alb.RandomBrightnessContrast(p=0.2),
                                                          alb.RandomGamma(p=0.2), 
                                                            alb.RGBShift(p=0.2), 
                                                          alb.VerticalFlip(p=0.5)], 
                                 bbox_params=alb.BboxParams(format='albumentations', 
                                                  label_fields=['class_labels']))

                                          def load_labels(label_path):
                                           with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
                                           label = json.load(f)
                                       return [label['class']], label['bbox']
- In real time detection 11.3 we successfully integrated the model into a camera system, allowing it to identify and localize individuals from the
dataset when they appear in the camera’s field of view
