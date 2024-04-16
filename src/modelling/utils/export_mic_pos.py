import numpy as np

# id, X-Z angle, X-Y angle 
ANGLE_DICT = { 1 : [ 69,  0],
               2 : [ 90, 32],
               3 : [111,  0],
               4 : [ 90,328],
               5 : [ 32,  0],
               6 : [ 55, 45],
               7 : [ 90, 69],
               8 : [125, 45],
               9 : [148,  0],
              10 : [125,315],
              11 : [ 90,291],
              12 : [ 55,315],
              13 : [ 21, 91],
              14 : [ 58, 90],
              15 : [121, 90],
              16 : [159, 89],
              17 : [ 69,180],
              18 : [ 90,212],
              19 : [111,180],
              20 : [ 90,148],
              21 : [ 32,180],
              22 : [ 55,225],
              23 : [ 90,249],
              24 : [125,225],
              25 : [148,180],
              26 : [125,135],
              27 : [ 90,111],
              28 : [ 55,135],
              29 : [ 21,269],
              30 : [ 58,270],
              31 : [122,270],
              32 : [159,271]}

def _get_pos_from_angles(theta, phi, radius = 4.2):
    
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    
    x = -radius * np.sin(theta) * np.cos(phi)
    y = -radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    return (x,y,z)

if __name__ == "__main__":
    coords = []
    for value in ANGLE_DICT.values():
        theta, phi = value
        coords.append(_get_pos_from_angles(theta, phi))
        
    coords = np.array(coords)
    ids = np.array(list(ANGLE_DICT.keys()))
    
    # save to file
    np.save("data/mic_pos.npy", coords)

    viz = False
    
    # ^^^^ set to True to visualize microphone positions
    # --------------------------------------------------
    
    if not viz:
        exit()
        
    # plot 3d scatter plot of microphone positions
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

    ax.scatter(coords[:,0], coords[:,1], coords[:,2])
    ax.set_box_aspect([1, 1, 1])  # Set equal aspect ratio for all axes

    # add labels in 3d space
    for i in range(1,33):
        ax.text(coords[i-1,0], coords[i-1,1], coords[i-1,2], f"{i}", color='black')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
