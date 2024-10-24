import tkinter as tk
import random

root=tk.Tk()
#X = int(input("Zadaj X:"))
#Y = int(input("Zadaj Y:"))
#print(X,Y)
max_x,max_y=1000,1000;
canvas=tk.Canvas(root, width=max_x, height=max_y);
canvas.pack()

array_ran_sur=[]

for i in range(20):
    ran_x=random.randrange(-5000,5000)
    ran_y=random.randrange(-5000,5000)
    ran_sur=[ran_x,ran_y]
    array_ran_sur.append(ran_sur)
    canvas.create_oval((ran_x//10)+500-2,(ran_y//10)+500-2,(ran_x//10)+500+2,(ran_y//10)+500+2, fill='black')
print(array_ran_sur)

def generate_more(arr, count):
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
    point[0]=point[0]+X_offset
    point[1]=point[1]+Y_offset
    arr.append(point)
    canvas.create_oval((point[0]//10)+500-2,(point[1]//10)+500-2,(point[0]//10)+500+2,(point[1]//10)+500+2, fill="red", outline="")
    #print(X_offset,Y_offset)
    return point

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

for count in range(10000):
    print(generate_more(array_ran_sur, count+1))
    #print(len(array_ran_sur))
kmeans_medoid(array_ran_sur, 5)
root.mainloop();
