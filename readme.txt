########## step 1 - environment setup #########
   	python 3.8.13
   	pytorch 1.12.1 py3.8_cuda11.3_cudnn8_0
	other require pkgs are list in requirements.txt
########## step 2 - self-defined testing variables setup #########

If you want to test the model with pretrained weight,set variable training in ./main.py line 61 to False
If you want train a new model,set variable training in main.py line 61 to True

----testing path------ 

	Set the test dataset path in ./main.py line 17 to your own directory path 	#test dataset input
	You can change the save path in  ./main.py line 43 or use the default path ./result 	#test dataset output

----testing filename------ 

	You can save result with the original filename or replace with the next line to save result with number,
	it can change in ./main.py line 57

	
########## step 3 - self-defined training variables setup #########
	
	Set the traing dataset path in ./trainer.py line 21 and 22 in numpy file to your own directory path 	#train dataset input
	The model weights save in ./weights/cvfsid_v2/ and change change in ./trainer.py line 90 and 92

########## step 4 - run the code #########

python main.py