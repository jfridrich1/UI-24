import tkinter as tk
import numpy as np
import random
import time

root=tk.Tk()
max_x_can,max_y_can=600,600;
canvas=tk.Canvas(root, width=max_x_can, height=max_y_can);
canvas.pack()

"""choice=int(input("Vyber metodu zhlukovania (1/2/3): "))
match choice:
    case 1:
        print("k means centroid")
    case 2:
        print("k metoid")
    case 3:
        print("divizne")"""

canvas.create_rectangle(50,50,550,550);

MAX_X, MAX_Y=5000,5000
MIN_X, MIN_Y=-5000,-5000

array_ran_sur=[]
num_points=20000
num_clus=20
radius=1
scaling_down=20
distances=np.zeros((num_points+20, num_clus))


def init_20(array,radius,scale):
    global MAX_X,MAX_Y,MIN_X,MIN_Y
    for i in range(20):
        ran_x=random.randrange(MIN_X,MAX_X)
        ran_y=random.randrange(MIN_Y,MAX_Y)
        ran_sur=[ran_x,ran_y]
        array.append(ran_sur)

    #print(array_ran_sur)


def generate_more(arr, count,radius,scale):
    point=random.choice(arr)

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
    return new

#------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
#vyber medoidov nahodne
def kmeans_draw(arr, k, radius, scale,centers_id):
    centers_sample=random.sample(arr,k)
    for num in range(k):
        centers_id.append(centers_sample[num])


# k-means++ inicializácia
def kmeans_plus_plus_init(arr, k):
    centers = []
    centers.append(random.choice(arr))  # Prvé centrum náhodne
    while len(centers) < k:
        distances = np.array([min([np.linalg.norm(np.array(point) - np.array(center)) ** 2 for center in centers]) for point in arr])
        probs = distances / distances.sum()
        cumprobs = np.cumsum(probs)
        r = random.random()
        for i, prob in enumerate(cumprobs):
            if r < prob:
                centers.append(arr[i])
                break
    return centers


#vypocet vzdialenosti bodov od medoidov, distances => maxtrix POINTS x MEDOIDS
def dist_calc(distances, array_centers, array_points):
    #print(f"centroidy v dist_calc: {centroids_id}")
    for i in range(len(array_points)):
        for j in range(len(array_centers)):
            medoid=np.array([array_centers[j][0],array_centers[j][1]])
            point=np.array([array_points[i][0],array_points[i][1]])
            distances[i,j]=round(np.linalg.norm(medoid-point),1)

#priradenie bodov ku klasterom
def clustering(distances, clusters, array):
    prirad=np.argmin(distances, axis=1)                     #axis=1 => prvy stlpec matrixu
    priradenie_zoznam=prirad.tolist()                       #premena na pouzitelne pole
    #print(f"toto je min: {priradenie_zoznam}")
    for point_index, cluster_index in enumerate(priradenie_zoznam):
        clusters[cluster_index].append(array[point_index])  #priradenie specifickeho bodu (nie index bodu) k polu specifickeho indexu klastera



def update_cent(clusters, iter):
    new_centroids = []
    for cluster in clusters:
        #print(f"klaster: {cluster} {len(cluster)}\n")
        if cluster:  # Skontroluj, či klaster nie je prázdny
            centroid = np.mean(cluster, axis=0).tolist()
            #print(centroid)
            new_centroids.append(centroid)
        """else:
            # Ak je klaster prázdny, pridaj pôvodný centroid
            new_centroids.append([0, 0])  # Môžeš nahradiť túto hodnotu inak, podľa potreby"""

    return new_centroids



def calculate_average_distance(clusters, centers):
    global array_worst_dist
    arr=[]
    cont = True
    #print(f"v clusters je {len(clusters)} klasterov")
    for c_index, cluster in enumerate(clusters):
        center = np.array(centers[c_index])
        distances = [np.linalg.norm(np.array(point)-center) for point in cluster]
        average_distance = np.mean(distances)
        #arr.append(average_distance)
        if average_distance > 500:
            array_worst_dist.append(average_distance)
            cont = False
            #return False  # Ak je priemerná vzdialenosť v niektorom klastri > 500, pokračuj v iterácii
    #print(arr)
    return cont  # Ak sú všetky klastre v poriadku, ukončíme cyklus


def draw_clusters(clusters):
    for cluster_index, cluster in enumerate(clusters):
        color = cluster_colors[cluster_index % len(cluster_colors)]
        for point in cluster:

            coord_x = (point[0] // scaling_down)+250+50
            coord_y = (point[1] // scaling_down)+250+50
            canvas.create_oval(coord_x - radius, coord_y - radius, coord_x + radius, coord_y + radius, fill=color, outline="")

        #print(cluster[cluster_index])
    for cluster_index, cluster in enumerate(clusters):
        color = cluster_colors[cluster_index % len(cluster_colors)]
        coord_x_center=(cluster[cluster_index][0] // scaling_down)+250+50
        coord_y_center=(cluster[cluster_index][1] // scaling_down)+250+50
        canvas.create_oval(coord_x_center-2, coord_y_center-2, coord_x_center+2, coord_y_center+2, fill=color, outline="black")


def kcent_clustering():
    global centroids_id, clusters, num_clus, array_ran_sur
    iteration_count = 0
    
    # Počiatočná inicializácia centroidov
    #centroids_id = random.sample(array_ran_sur, num_clus)
    centroids_id = kmeans_plus_plus_init(array_ran_sur, num_clus)
    
    while iteration_count < 20:
        iteration_count+=1
        print(f"\nIterácia: {iteration_count}")
        
        # Krok 1: Vypočítaj vzdialenosti medzi každým bodom a centroidmi
        dist_calc(distances, centroids_id, array_ran_sur)
        
        # Krok 2: Priraď body k najbližším centroidom
        clusters = [[] for _ in range(num_clus)]
        clustering(distances, clusters, array_ran_sur)

        # Krok 3: Aktualizácia centroidov
        #print(centroids_id)
        new_centroids = update_cent(clusters, iteration_count)
        #print(new_centroids)
        
        # Skontroluj, či sa centroidy zmenili; ak nie, ukonči cyklus
        """if new_centroids == centroids_id:
            print("Centroidy sa stabilizovali.")
            break"""
        
        centroids_id = new_centroids  # Aktualizácia centroidov pre ďalšiu iteráciu

        # Krok 4: Skontroluj priemerné vzdialenosti (voliteľné)
        if calculate_average_distance(clusters, centroids_id):
            print("Dosiahnuty ciel")
            break  # Ukonči cyklus, ak sú všetky priemerné vzdialenosti v poriadku
    draw_clusters(clusters)




#----------------------------------------------------------------------------------------------------------------------------------------
centroids_id=[]
medoids_id=[]

new_medoids=[]
new_centroids=[]

array_worst_dist=[]

cluster_colors = [
    "red", "blue", "green", "purple", "orange", "yellow", "pink", "cyan", "magenta", "brown",
    "lime", "indigo", "teal", "gold", "lightblue", "coral", "darkgreen", "salmon", "violet", "khaki"
]

start=time.time()
#inicializacia prvych 20 bodov a dodatocnych 40k bodov
init_20(array_ran_sur,radius,scaling_down);
for count in range(num_points):
    generate_more(array_ran_sur, count+1,radius,scaling_down)
num_points+=20

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#kmeans_draw(array_ran_sur,num_clus,radius,scaling_down,centroids_id)
kcent_clustering()
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

end=time.time()
print(f"cas bezania: {end-start}")
root.mainloop();






"""
def update_med(new_medoids, clusters, color, iter):
    #print("update funkcia")
    radius=1
    for cluster in clusters:
        distances_in_cluster=[]
        for point in cluster:
            total_sum=sum(np.linalg.norm((np.array(point))-np.array(other_point)) for other_point in cluster)
            distances_in_cluster.append((point, total_sum))

        new_medoid=min(distances_in_cluster, key=lambda x: x[1])[0]
        new_medoids.append(new_medoid)
        #print(f"{new_medoids}")

        if iter==10:
            color="cyan"
            coord_x=(new_medoid[0]//20)+250+50   #+50 kvoli okraju
            coord_y=(new_medoid[1]//20)+250+50
            canvas.create_oval(coord_x-radius, coord_y-radius, coord_x+radius, coord_y+radius, width=5, outline=color)
    
    #print("update done")
    return new_medoids
    
    
    
    
    
    # Hlavný cyklus
def kmed_clustering():
    global medoids_id, new_medoids, clusters, num_clus, num_points, array_worst_dist

    iteration_count = 0
    color="black"
    while iteration_count!=20:
        iteration_count += 1
        print(f"\nIterácia: {iteration_count}")

        # Krok 1: Vypočítaj vzdialenosti a priraď body k najbližším medoidom
        dist_calc(distances, medoids_id, array_ran_sur)
        clusters = [[] for _ in range(num_clus)]  # Reset klastra pre novú iteráciu
        clustering(distances, clusters, array_ran_sur)

        # Krok 2: Aktualizuj medoidy
        new_medoids = []
        update_med(new_medoids, clusters, color, iteration_count)
        medoids_id = new_medoids

        # Krok 3: Skontroluj priemerné vzdialenosti
        if calculate_average_distance(clusters, medoids_id):
            #print("Podmienka splnená: Priemerná vzdialenosť v každom klastri je ≤ 500.")
            break  # Ukonči cyklus, ak sú všetky priemerné vzdialenosti v poriadku
        #else:
            #print("Podmienka nesplnená: Opakujeme klastrovanie.")
    """