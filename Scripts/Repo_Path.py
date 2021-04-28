#-------------------------------------------------------------------------------------------
#  You might want to use this script if you run into path issues 
#-------------------------------------------------------------------------------------------

import os

repo_path = os.getcwd() if os.getcwd().split("\\")[-1] == "Repository" else os.path.dirname(os.getcwd())

print(repo_path)