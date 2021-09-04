import os
import time

specificFile = 'words20to25'
start = time.time()
shutdown = input("Do you wish to shutdown your computer ? (yes / no): ")
turnOff = True

if shutdown == 'yes':
    f = open("./Generated Files/PREDwords (NC).txt", "a", encoding='utf-8')
    f.write("Test")