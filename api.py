from flask import Flask, render_template, request,Response
from flask_cors import CORS
import cv2,imutils,time
import numpy as np
import pyshine as ps
import os
app = Flask(__name__)
CORS(app, support_credentials=True)
@app.route('/')
def index():
   return render_template("image-detection.html")
   
def Day(img):
    return img
       
def Night(img):
   
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r_image, g_image, b_image = cv2.split(img)
    r_image_eq = cv2.equalizeHist(r_image) 
    g_image_eq = cv2.equalizeHist(g_image)
    b_image_eq = cv2.equalizeHist(b_image)
    img = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    im_blurred = cv2.GaussianBlur(img, (1,1), 1)
    img = cv2.addWeighted(img, 1 + 3.0, im_blurred, -3, 0) 
    return img
def detection(params):
    print(params)
    count=0
    count1=0
    count2=0
    count3=0
    count4=0
    count5=0
    count6=0
    count7=0
    count8=0
    count9=0
    count10=0
    count11=0
    count12=0
    count13=0
    count14=0
    count15=0
    count16=0
    count17=0
    count18=0
    count19=0
    count20=0
    
    cap= cv2.VideoCapture('105.mp4') 
    #address="http://10.212.187.181:8080/video"
    #cap.open(address)
        
    net = cv2.dnn.readNetFromDarknet("yolov4-tiny_custom.cfg","yolov4-tiny_custom_final.weights")
    classes = ['','bus','rickshaw','motorbike','car','three wheelers (CNG)','pickup','minivan','suv','van','truck','bicycle','policecar','ambulance','human hauler','wheelbarrow','minibus','auto rickshaw','army vehicle','scooter','garbagevan']   
    while(1):
    
        ret, img = cap.read()
        file = str(params["file"])
        if file == "Day" :
           img = Day(img)
        else:
           img = Night(img)
        
        hight,width= img.shape[0:2]
        img[0:140,0:width]=[255,255,255]
        cv2.line(img,(0,hight-96),(width,hight-96),(0,225,225),1)

        blob = cv2.dnn.blobFromImage(img, 1/255,(448,448),(0,0,0),swapRB = False,crop= False)

        net.setInput(blob)

        output_layers_name = net.getUnconnectedOutLayersNames()

        layerOutputs = net.forward(output_layers_name)
    
   

               
    
    
        boxes =[]
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * hight)
                    w = int(detection[2] * width)
                    h = int(detection[3]* hight)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x,y,w,h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
                
                
                   

        indexes = cv2.dnn.NMSBoxes(boxes,confidences,.3,.3)
    
       
        font = cv2.FONT_HERSHEY_PLAIN
        if  len(indexes)>0:
          
            for i in indexes.flatten():
                label = str(classes[class_ids[i]])
                x,y,w,h = boxes[i]
                veh=y+h/2
                
                lineveh=hight-96
                
               
                if((veh<lineveh+3.2 and veh>lineveh-3.2)and label == "bus"):
                   count = count + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+2.5 and veh>lineveh-2.5)  and label == "car"):
                    count1=count1+1
                    cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                   
                elif((veh<lineveh+3.2 and veh>lineveh-3.2)and label == "three wheelers (CNG)"):
                   count2 = count2 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+3.2 and veh>lineveh-3.2)and label == "rickshaw"):
                   count3= count3 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+3.2 and veh>lineveh-3.2)and label == "truck"):
                   count4= count4 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),3.2)
                elif((veh<lineveh+2.5 and veh>lineveh-2.5)and label == "motorbike"):
                   count5= count5 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+3.2 and veh>lineveh-3.2)and label == "pickup"):
                   count6= count6 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+3.2 and veh>lineveh-3.2)and label == "minivan"):
                   count7= count7 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+4 and veh>lineveh-4)and label == "suv"):
                   count8= count8 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+4 and veh>lineveh-5)and label == "van"):
                   count9= count9 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+2 and veh>lineveh-2)and label == "taxi"):
                   count10= count10 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+5 and veh>lineveh-5)and label == "bicycle"):
                   count11= count11 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+5 and veh>lineveh-5)and label == "policecar"):
                   count12= count12 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+5 and veh>lineveh-5)and label == "ambulance"):
                   count14= count14 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+5 and veh>lineveh-5)and label == "human hauler"):
                   count14= count14 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+5 and veh>lineveh-5)and label == "wheelbarrow"):
                   count15= count15 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+5 and veh>lineveh-5)and label == "minibus"):
                   count16= count16 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+5 and veh>lineveh-5)and label == "auto rickshaw"):
                   count17= count17 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+5 and veh>lineveh-5)and label == "army vehicle"):
                   count18= count18 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+5 and veh>lineveh-5)and label == "scooter"):
                   count19= count19 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                elif((veh<lineveh+5 and veh>lineveh-5)and label == "garbagevan"):
                   count20= count20 + 1
                   cv2.line(img,(0,hight-96),(width,hight-96),(0,0,96),4)
                confidence = str(round(confidences[i],2))
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
                cv2.putText(img,label + " " + confidence, (x,y+20),font,1,(0,0,255),1) 
            
                    
        
        
        if ret == True:
           
           img = cv2.resize(img,(832,680))
           
           text1 ='bus:'+  str(count)
           img=ps.putBText(img,text1,text_offset_x=10,text_offset_y=5,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255))
           text2 ='car:'+  str(count1)
           img=ps.putBText(img,text2,text_offset_x=75,text_offset_y=5,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text3 ='CNG:'+  str(count2)
           img=ps.putBText(img,text3,text_offset_x=155,text_offset_y=5,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text4 ='rickshaw:'+  str(count3)
           img=ps.putBText(img,text4,text_offset_x=235,text_offset_y=5,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text5 ='truck:'+  str(count4)
           img=ps.putBText(img,text5,text_offset_x=360,text_offset_y=5,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text6 ='m.bike:'+  str(count5)
           img=ps.putBText(img,text6,text_offset_x=450,text_offset_y=5,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text7 ='pickup:'+  str(count6)
           img=ps.putBText(img,text7,text_offset_x=560,text_offset_y=5,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text8 ='minivan:'+  str(count7)
           img=ps.putBText(img,text8,text_offset_x=665,text_offset_y=5,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text9 ='suv:'+  str(count8)
           img=ps.putBText(img,text9,text_offset_x=750,text_offset_y=50,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text10 ='van:'+  str(count9)
           img=ps.putBText(img,text10,text_offset_x=5,text_offset_y=28,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text11 ='taxi:'+  str(count10)
           img=ps.putBText(img,text11,text_offset_x=85,text_offset_y=28,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text12 ='bicycle:'+  str(count11)
           img=ps.putBText(img,text12,text_offset_x=165,text_offset_y=28,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text13 ='policeC.:'+  str(count12)
           img=ps.putBText(img,text13,text_offset_x=270,text_offset_y=28,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text14 ='ambulance:'+  str(count13)
           img=ps.putBText(img,text14,text_offset_x=380,text_offset_y=28,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text15 ='H.hauler:'+  str(count14)
           img=ps.putBText(img,text15,text_offset_x=520,text_offset_y=28,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text16 ='w.barrow:'+  str(count15)
           img=ps.putBText(img,text16,text_offset_x=635,text_offset_y=28,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text17 ='minibus:'+  str(count16)
           img=ps.putBText(img,text17,text_offset_x=600,text_offset_y=50,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text18 ='a.rickshaw:'+  str(count17)
           img=ps.putBText(img,text18,text_offset_x=5,text_offset_y=50,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text19 ='army vehicle:'+  str(count18)
           img=ps.putBText(img,text19,text_offset_x=150,text_offset_y=50,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text20 ='scooter:'+  str(count19)
           img=ps.putBText(img,text20,text_offset_x=320,text_offset_y=50,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           text21 ='g.van:'+  str(count20)
           img=ps.putBText(img,text21,text_offset_x=430,text_offset_y=50,vspace=3,hspace=3, font_scale=0.6,background_RGB=(255,255,255),text_RGB=(1,1,1))
           
               
           
           #text = str(time.strftime("%d %b %Y %H. %M. %S %p"))
           #img=ps.putBText(img,text,text_offset_x=255,text_offset_y=15,background_RGB=(228,20,132))
           text = str(time.strftime("%d %b %Y "))
           pred_path = os.path.join(r"C:\Users\rafi\Desktop\New folder (4)",text+".jpg")
           cv2.imwrite(pred_path, img)
           pred_path = os.path.join(r"C:\Users\rafi\Desktop\New folder (4)",text+".txt")
           with open(text+".txt", "w") as outfile:
               outfile.write(text1)
               outfile.write("\n")
               outfile.write(text2)
               outfile.write("\n")
               outfile.write(text3)
               outfile.write("\n")
               outfile.write(text4)
               outfile.write("\n")
               outfile.write(text5)
               outfile.write("\n")
               outfile.write(text6)
               outfile.write("\n")
               outfile.write(text7)
               outfile.write("\n")
               outfile.write(text8)
               outfile.write("\n")
               outfile.write(text9)
               outfile.write("\n")
               outfile.write(text10)
               outfile.write("\n")
               outfile.write(text11)
               outfile.write("\n")
               outfile.write(text12)
               outfile.write("\n")
               outfile.write(text13)
               outfile.write("\n")
               outfile.write(text14)
               outfile.write("\n")
               outfile.write(text15)
               outfile.write("\n")
               outfile.write(text16)
               outfile.write("\n")
               outfile.write(text17)
               outfile.write("\n")
               outfile.write(text18)
               outfile.write("\n")
               outfile.write(text19)
               outfile.write("\n")
               outfile.write(text20)
               outfile.write("\n")
               outfile.write(text21)
               outfile.write("\n")
         
               
               outfile.close()
              
           frame = cv2.imencode(".jpg",img)[1].tobytes()
           yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'  + frame+ b'\r\n')
           time.sleep(0.1)
        else:
           break
        

@app.route('/res',methods = ['POST','GET'])
def res():
    global result
    if request.method =='POST':
         result = request.form.to_dict()
         return render_template("results.html",result=result)


@app.route('/video_feed')
def video_feed():
    global result
    params=result
    return Response(detection(params),mimetype = "multipart/x-mixed-replace;boundary=frame")

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")