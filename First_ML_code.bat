powershell "Start-Process cmd -Verb RunAs -ArgumentList '/c conda init cmd.exe & exit'"
start /min cmd /k "cd C:\Users\harsh\OneDrive\Desktop\My projects & conda activate detectron2 & python ML1.py"


