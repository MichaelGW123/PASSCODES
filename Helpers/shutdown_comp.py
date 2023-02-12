import os

run_shutdown = False

if run_shutdown:
    os.system("shutdown /s /t 1")
else:
    print("Not shutdown")
