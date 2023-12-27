#pip install -r requirements.txt

# for virtual environment 
# python -m venv venv
#.\venv\Scripts\activate
#pip install -r requirements.txt



import utils, csv 


if __name__ == "__main__":
    names = ['1.2.jpg','2.2.jpg','3.2.jpg','4.2.jpg','5.2.jpg']
    #names = ['5.2.jpg']
    csv_file_path = 'output.csv'

    for name in names:

        qr,row = utils.rowSplit(name)
        l1 = utils.ansByHorizontalSplit(row)
        l2 = utils.ansByCricleRadious(row)

        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([qr,l1])
        
        """
        print(f'                                           {name}')
        for i in range(0,70):
            if l1[i]!=l2[i]:
                print(f'                                           ({i+1})')
                utils.print_img(row[i])
                print(l1[i],l2[i])
        """