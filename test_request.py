import requests, time

HOST = '127.0.0.1'
PORT = '8002'

image_path = r"/home/hungdv/tcgroup/Clone_project/table_reconstruction2/img_test/img_test.jpg"

def htmlTableAPI(img_path):
    files = {"image": open(img_path, 'rb')}
    resp = requests.post(f'http://{HOST}:{PORT}/extract-table-trace', files=files)
    
    if resp.status_code == 200:
        data_recv = resp.json()
        return data_recv
    else:
        return None

# Return [x1, y1, x2, y2 ,x3, y3, x4, y4]
#  
#   x1,y1 ________________________ x4,y4
#        |                        |
#        |                        |
#        |                        |
#   x2,y2|________________________|x3,y3

t = time.time()
table = htmlTableAPI(image_path)
print(time.time()-t)
try:
    f = open("table.html", "w")
    f.write(table["table"])
    f.close()

    print(table["table"])
except:
    print(table)

