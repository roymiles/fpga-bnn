# Final-year-project2

Compile the program using the create_demo.sh script with either hw_nolib, hw_lib or sw as the parameters to compile the program from source using the hardware bnn implementation, using precompiled libraries, or using the HLS BNN implementation.

Once compiled, the program can be run "./program_nolib --args"

## NOTE
*The respective authors and my modifications have been noted to the top of every relevant top level file (inside app/)*
  
 The setup is discussed in further detail in my Thesis.
 
 
 Files of interest:
 1. app/app.cpp
 2. app/app.h
 3. app/common.h
 4. app/creat_demo.sh
 5. app/foldedmv-offload.cpp
 6. app/foldedmv-offload.h
 7. app/main_python.cpp
 8. app/opencv_utils.cpp
 9. app/opencv_utils.h
 10. app/xblackboxjam_hw.h
 11. MotionSegmentation.cpp
 12. All inside boot_files/


The GitHub repository can be found at: https://github.com/iyop45/Final-year-project
A lot of the development code has not been uploaded as it will only act as bloat. 
For example, there is over 10GB of boot files saved across the development cycle for various accelerator designs and lots of other source files for the drivers and testing.
Generally, the framework was built in a iterative manner with lots of modular unit testing blocks to ensure reliability across all versions.

If there are any questions about how to use this framework or how to add to it, let me know at rm14075@bristol.ac.uk
