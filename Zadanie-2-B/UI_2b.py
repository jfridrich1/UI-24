import tkinter as tk
import numpy as np
import random

MAX_X, MAX_Y=5000,5000
MIN_X, MIN_X=-5000,-5000

root=tk.Tk()
max_x_can,max_y_can=600,600;
canvas=tk.Canvas(root, width=max_x_can, height=max_y_can);
canvas.pack()

canvas.create_rectangle(50,50,550,550);


array_ran_sur=[]
num_points=20
num_clus=5
radius=1
scaling_down=20
distances=np.zeros((40, 5))


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


def kmeans_medoid(arr, k, radius, scale,medoids_id):
    medoids_sample=random.sample(arr,k)
    for num in range(k):
        medoids_id.append(medoids_sample[num])
    print(f"zoznam medoidov: {medoids_id}")
    for i in range(k):
        coord_x=(medoids_sample[i][0]//scale)+250+50   #+50 kvoli okraju
        coord_y=(medoids_sample[i][1]//scale)+250+50
        canvas.create_oval(coord_x-radius,coord_y-radius,coord_x+radius,coord_y+radius, width=5)

def dist_calc(distances, array_m, array_points):
    print(len(array_points))
    print(len(array_m))
    for i in range(len(array_points)):
        for j in range(len(array_m)):
            medoid=np.array([array_m[j][0],array_m[j][1]])
            point=np.array([array_points[i][0],array_points[i][1]])
            print("--------------------")
            print(array_points[i])
            print(array_m[j])
            distances[i,j]=round(np.linalg.norm(medoid-point),2)
            print(distances[i,j])

def div_cluster():
    pass




init_20(array_ran_sur,radius,scaling_down);
for count in range(num_points):
    print(generate_more(array_ran_sur, count+1,radius,scaling_down))
num_points+=20

medoids_id=[]
kmeans_medoid(array_ran_sur,num_clus,radius,scaling_down,medoids_id)
dist_calc(distances, medoids_id, array_ran_sur)
root.mainloop();
