# Attendance-Project
The design and creation of a smart attendance system.


Using single Image of Overall class Students and marking students as Present or Absent using Keras.Facenet model
step 1 : LOGIN using (username : admin , password : 0000)
step 2 : Upload a single image contains overall class students at that time, click submit and the model will predict and click extract to see the markings
step 3 : Extraction of Markings in a tableView
to change the training Image , Create a Folder inside Train_images and Add image into that created individual folder for students
to Train the image , open face_trainer_mtcnn.ipynb and save the datapoints using pickle
and Run the app.py from main/app.py
Check for Directories inside every file, change it according to your Sys directory
