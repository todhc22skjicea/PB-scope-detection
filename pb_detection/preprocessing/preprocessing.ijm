// MACRO DESCRIPTION:  
// This macro is designed to process image files for quantification,   
// and save the processed images in a designated output directory.  
// PARAMETERS:  
// - source_path: The path of raw image data.  
// - output_path: The path where the processed images will be saved.  
// - Threshold values: Set to 120 (min) and 330 (max) for the quantification analysis.  
// - Image conversion: Images are converted to 8-bit format.  
// - File format for saving: tif

// DIRECTORY STRUCTURE:  
// - Raw data is assumed to be stored in subdirectories labeled "T1" to "T8" (representing different trials or time points),   
//   each containing further subdirectories labeled "G1" to "G5" (representing different groups or conditions).  
// - Processed images will be saved in a corresponding directory structure under the defined output path.  
// 
// - dataraw/
//   └──T1/ 
//   │  ├── G1/
//	 │	│	├── pbodys/ 
//   │  │	│	├── image1.tif: 
//   │  │	│	├── image2.tif: 
//	 │	│	│	└──...
//   │ 	├── G2/ 
//	 │	│	├── pbodys/ 
//   │  │	│	├── image1.tif: 
//	 │	│	│	└──...
//   └──T2/ 
//   │ 	├── G1/
//   │	│	├── pbodys/ 	
//	 │	│	│	└──...
//   │  ├── G2/ 
//	 │	│	└──...
//   ......
//   ......


//run("Brightness/Contrast...");
source_path = "D:\\result\\dataraw"
output_path = "D:\\result\\quantify"
for(i=1;i<9;i++)
{
	for (j=1;j<6;j++)
	{
		directory = source_path + "\\T" + i + "\\G" + j + "\\pbody\\";
		output_directory = output_path + "\\T" + i + "\\G" + j + "\\pbody\\" ;
		if (!File.exists(output_directory)) {  
	                File.makeDirectory(output_directory);  
        }  
		list = getFileList(directory);
		for(k = 0;k<list.length;k++)
		{
			open(directory+list[k]);
			//quantify analysis threshold
			setMinAndMax(450, 1000);
			setOption("ScaleConversions", true);
			run("8-bit");
			saveAs("tif", output_directory + getTitle);
			close();
		}
	}
}
