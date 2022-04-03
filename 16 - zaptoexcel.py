import pandas as pd

#colocar o arquivo exportado do whatsapp no mesmo diretório
file_path = "zap.txt"

with open(file_path, mode='r', encoding="utf8") as f:
    data = f.readlines()


dataset = data[1:]
cleaned_data = []

for line in dataset:
    # Check, whether it is a new line or not
    # If the following characters are in the line -> assumption it is NOT a new line
    if '/' in line and ':' in line and ' ' in line and '-' in line:
        # grab the info and cut it out
        date = line.split(" ")[0]
        line2 = line[len(date):]
        time = line2.split("-")[0][2:]
        line3 = line2[len(time):]
        name = line3.split(":")[0][4:]
        line4 = line3[len(name):]
        message = line4[6:-1] # strip newline charactor
        cleaned_data.append([date, time, name, message])
# else, assumption -> new line. Append new line to previous 'message'
    else:
        new = cleaned_data[-1][-1] + " " + line
        cleaned_data[-1][-1] = new

# Create the DataFrame
df = pd.DataFrame(cleaned_data, columns = ['Date', 'Time', 'Name', 'Message'])




# Save it!
df.to_excel('chat_history1.xlsx', index=False)

#adaptado de : https://github.com/Sven-Bo/WhatsApp-History-Export-Excel/blob/master/convert_to_excel.py

