## CVPR2021 NTIRE 2021 Depth Guided Relighting Challenge

In this repo, we include the 1st Place code and submission to [NTIRE 2021 Depth Guided Relighting Challenge Track 2: Any-to-any relighting](https://competitions.codalab.org/competitions/28031) and submission to [NTIRE 2021 Depth Guided Relighting Challenge Track 1: One-to-one relighting](https://competitions.codalab.org/competitions/28030) of team DeepBlueAI


The submission for track1 can be find at ![](https://github.com/Qidian213/NTIRE_Relighting/tree/main/Meida/Test_Track1_DeepBlueAI)

The submission for track2 can be find at ![](https://github.com/Qidian213/NTIRE_Relighting/tree/main/Meida/Test_Track2_DeepBlueAI)


### Performance：
 NTIRE 2021 Depth Guided Relighting Challenge Leaderboard
![](https://github.com/Qidian213/NTIRE_Relighting/blob/main/Meida/Score_Result.png)

 And Some Visible Results
![](https://github.com/Qidian213/NTIRE_Relighting/blob/main/Meida/Image_Result.png)

### NetWork:
![](https://github.com/Qidian213/NTIRE_Relighting/blob/main/Meida/Network.jpg)

### Train and Test
	1, Dependencies
		 Python  3.8
		 Pytorch 1.6

	2, Training 
		 a. change the dataset path in file 'Options/options.py'
		 b. run cmd : python Train_Guide.py , all result will be find in 'work_space' dir 

	3, Testing For track1 
		 a. change 'input_folder' to test dataset dir,  line 97 in file 'Track1_Test_Ensemble.py'
		 b. run cmd : python Track1_Test_Ensemble.py , result will be find in 'work_space/Test_Track1_DeepBlueAI' dir 

	4, Testing For track2 
		 a. change 'input_folder'  and 'guide_folder' to test dataset dir,  line 99-100 in file 'Track2_Test_Ensemble.py'
		 b. run cmd : python Track2_Test_Ensemble.py, result will be find in 'work_space/Test_Track2_DeepBlueAI' dir 
		 
