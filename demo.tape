# Where should we write the GIF?
Output demo.gif

# Set up a 1200x600 terminal with 46px font.
Set FontSize 24
Set Width 1440
Set Height 2560
Set WindowBar Colorful
Set LoopOffset 5 # Start the GIF at the 5th frame
Set Framerate 6
Set TypingSpeed 15ms

# Type a command in the terminal.
Type "python -m accelerate.commands.launch --main_process_port=12350 --num_processes=8 lmms_eval --model=llava --model_args=pretrained=liuhaotian/llava-v1.5-7b --tasks=mme --limit=8 --batch_size=1 --log_samples --log_samples_suffix=demo --output_path=./logs/"
Enter
# Admire the output for a bit.
Sleep 30
