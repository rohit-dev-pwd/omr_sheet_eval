import utils, os, csv

current_directory = os.getcwd()
omr_files = [file for file in os.listdir(current_directory) if file.endswith('.jpg') and file.startswith('d')]
omr_top_files = [file for file in os.listdir(current_directory) if file.endswith('.jpg') and file.startswith('t')]


csv_output = 'output.csv'
csv_setcode = 'setcode.csv'
csv_anskey = 'anskey.csv'
csv_mark = 'marks.csv'

utils.omrTocsv(omr_files,csv_output)
utils.omrTopTocsv(omr_top_files, csv_setcode)  

data = utils.getMarkList(csv_output)
qrWithSet = utils.getqrRollSet(csv_setcode)
qrSetandres = utils.getAnskey(csv_anskey)

utils.generateResultSheet(csv_mark,data,qrWithSet,qrSetandres)