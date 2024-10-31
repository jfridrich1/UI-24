import tkinter as tk
import random

MAX_X, MAX_Y=5000,5000
MIN_X, MIN_X=-5000,-5000

root=tk.Tk()
max_x_can,max_y_can=600,600;
canvas=tk.Canvas(root, width=max_x_can, height=max_y_can);
canvas.pack()

canvas.create_rectangle(50,50,550,550);

array_ran_sur=[]
radius=1
scaling_down=20

def init_20(array,radius,scale):
    for i in range(20):
        ran_x=random.randrange(-5000,5000)
        ran_y=random.randrange(-5000,5000)
        ran_sur=[ran_x,ran_y]
        array.append(ran_sur)

        coord_x=(ran_x//scale)+250+50   #+50 kvoli okraju
        coord_y=(ran_y//scale)+250+50
        canvas.create_oval(coord_x-radius,coord_y-radius,coord_x+radius,coord_y+radius, fill='black')
    #print(array_ran_sur)


def generate_more(arr, count,radius,scale):
    point=random.choice(arr)

    print(f"Generovanie cisla c.{count}")
    #print(point)

    #zredukovat interval!!!!
    X_offset=random.randint(-100,100)
    Y_offset=random.randint(-100,100)
    
    while point[0]+X_offset>5000 or point[0]+X_offset<-5000 or point[1]+Y_offset>5000 or point[1]+Y_offset<-5000:
        if point[0]+X_offset>5000 or point[0]+X_offset<-5000:
            X_offset=random.randint(-100,100)
        elif point[1]+Y_offset>5000 or point[1]+Y_offset<-5000:
            Y_offset=random.randint(-100,100)
        else:
            continue

    new_x=point[0]+X_offset
    new_y=point[1]+Y_offset
    new=[new_x,new_y]

    arr.append(new)

    coord_x=(new[0]//scale)+250+50   #+50 kvoli okraju
    coord_y=(new[1]//scale)+250+50

    canvas.create_oval(coord_x-radius,coord_y-radius,coord_x+radius,coord_y+radius, fill="red", outline="")
    return new

def kmeans_centroid():
    pass
def kmeans_medoid(arr, k):
    medoids_id=random.sample(arr,k)
    print(medoids_id)
    for i in range(k):
        print(i)
        canvas.create_oval((medoids_id[i][0]//10)+500-5,(medoids_id[i][1]//10)+500-5,(medoids_id[i][0]//10)+500+5,(medoids_id[i][1]//10)+500+5, width=5)

def div_cluster():
    pass




init_20(array_ran_sur,radius,scaling_down);
for count in range(10000):
    print(generate_more(array_ran_sur, count+1,radius,scaling_down))
kmeans_medoid(array_ran_sur, 5)
root.mainloop();
