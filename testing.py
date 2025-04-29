import numpy as np
import time
import re
import skimage as ski
import os
import sys
import platform

"""def bresenham(x0: int, y0: int, x1: int, y1: int):
    
    Bresenham's Line Generation Algorithm
    https://www.youtube.com/watch?v=76gp2IAazV4
    
    print ("START From" + str(x0) + ", " + str(y0) + ", " + str(x1) + ", " + str(y1) + ".")  
    # step 1 calculate difference
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    mm = False
    if (dx == 0):
        m = 999999
        mm = True
    else:
        m = dy/dx
     
    print (str(dy) + " > " + str(dx))
    if dy > dx:
        print("Y Increment")
        if (y0 > y1):
            print("Inverted")
            return bresenham(x1, y1, x0, y0)
    else:
        print("X Increment")
        if (x0 > x1):
            print("Inverted")
            return bresenham(x1, y1, x0, y0)

    # step 2 perform test to check if pk < 0
    flag = True
    
    line_pixel = []
    line_pixel.append([x0,y0])
    
    step = 1
    if x0>x1 or y0>y1:
        step = -1

      
    if m < 1:
        x0, x1 ,y0 ,y1 = y0, y1, x0, x1
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        mm = True

    print ("ITERATE From" + str(x0) + ", " + str(y0) + ", " + str(x1) + ", " + str(y1) + ".")  
        
    p0 = 2*dx - dy
    x = x0
    y = y0
    
    for i in range(abs(y1-y0)):
        if flag:
            x_previous = x0
            p_previous = p0
            p = p0
            flag = False
        else:
            x_previous = x
            p_previous = p
            
        if p >= 0:
            x = x + step

        p = p_previous + 2*dx -2*dy*(abs(x-x_previous))
        y = y + 1
        
        if mm:
            if x0>x1:
                line_pixel.append([-y,-x])
            else:
                line_pixel.append([y,x])
        else:
            if y0>y1:
                line_pixel.append([-x,-y])
            else:
                line_pixel.append([x,y])
            
            
    return line_pixel
def bresenham(x1: int, y1: int, x2: int, y2: int):
    dx: int = x2 - x1
    dy: int = y2 - y1
    dy2: int = dy * 2
    dx2: int = dx * 2
    pk: int = dy2 - dx

    if abs(dx) > abs(dy):
        steps: int = abs(dx)
    else:
        steps: int = abs(dy)

    xcoordinates = []
    ycoordinates = []

    i = 0
    while i < steps:
        i += 1
        if pk >= 0:
            pk = pk + dy2 - dx2
            x1 = x1 + 1
            y1 = y1 + 1
            print("x1: ", x1, "y1:", y1)
            xcoordinates.append(x1)
            ycoordinates.append(y1)
        else:
            pk = pk + dy2
            x1 = x1 + 1
            y1 = y1
            print("x1: ", x1, "y1:", y1)
            xcoordinates.append(x1)
            ycoordinates.append(y1)
    return [xcoordinates, ycoordinates]

    m_new: int = 2 * (y2 - y1)
    slope_error_new: int = m_new - (x2 - x1)
    x: int = x1
    y: int = y1
    while x <= x2: 
        print ("(" + x + "," + y + ")\n") 
  
        # Add slope to increment angle formed 
        slope_error_new += m_new; 
  
        # Slope error reached limit, time to 
        # increment y and update slope error. 
        if (slope_error_new >= 0):
            y += 1; 
            slope_error_new -= 2 * (x2 - x1); 
        x+=1"""

def isect_tri_plane_v3(p0, p1, p2, p_co, p_no):
    lines = [[p0, p1], [p1, p2], [p0, p2]]
    isect = []
    for line in lines:   
        isect.append(isect_line_plane_v3(line[0], line[1], p_co, p_no, 0.001))
    return list(filter(lambda x: x is not None, isect)) 
  
# intersection function
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        finalpoint = add_v3v3(p0, u)
        if is_point_on_line_v3(finalpoint, p0, p1):
            return finalpoint
        else:
            return None

    # The segment is parallel to plane.
    return None

def is_point_on_line_v3(point, linePoint0, linePoint1, epsilon=0.01):
    lineVector = sub_v3v3(linePoint1, linePoint0)
    pointVector = sub_v3v3(point, linePoint0)
    if (magnitude_v3(pointVector) < magnitude_v3(lineVector)): # and (dot_v3v3(lineVector, pointVector) < epsilon)):
        return True
    else:
        return False

# ----------------------
# generic math functions

def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
    )

def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
    )

def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
    )

def len_squared_v3(v0):
    return dot_v3v3(v0, v0)

def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
    )
    
def normalize_v3(vec):
    magnitude = magnitude_v3(vec)
    return [vec[0]/magnitude, vec[1]/magnitude, vec[2]/magnitude]
    
def magnitude_v3(vec):
    return np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

def euler_to_dir(vec, mag = 1):
    x = 1
    z = np.tan(vec[1])
    newvec = [x, 0, z]
    newvec =  normalize_v3(newvec)
    newvec[1] = np.sin(vec[0])*np.cos(vec[2])
    newvec = mul_v3_fl(newvec, mag)

    return (newvec)

def dir_to_euler(vec):
    vec = normalize_v3(vec)
    pitch=np.asin(vec[1])
    if vec[2] == 0:
        if vec[0] > 0:
            yaw = 90
        else:
            yaw = -90
    else:
        yaw = np.atan(vec[0]/vec[2])
    roll = 0
    return ([float(pitch), float(yaw), roll])

def matrixRotation(vec, y_offset):
    x, y, z = vec[0], vec[1], vec[2]
    x2 = x*(np.cos(y_offset)) - z*(np.sin(y_offset))
    y2 = y
    z2 = x*(np.sin(y_offset)) + z*(np.cos(y_offset))
    return [float(x2), float(y2), float(z2)]

def align_vector(vec, y_offset):
    """magnitude = magnitude_v3(vec)
    print ("Magnitude is " + str(magnitude))
    angles = dir_to_euler(vec)
    print ("Eulers begin at " + str(angles))
    angles[1] -= y_offset
    print ("Eulers are augmented at " + str(angles))
    direction = euler_to_dir(angles, magnitude)
    print ("Eulers become directional again at " + str(direction))"""
    return matrixRotation(vec, np.deg2rad(y_offset))

# A = int(input('Enter 1st number: '))
# b = int(input('Enter 2nd number: '))

# print(f'Sum of {a} and {b} is {sum(a, b)}')``
p_co = [0, 0, 0]
p_no = [0, 0, 1]
print ("planar angles are: " + str(dir_to_euler(p_no)))

if platform.system() == "Linux":
    filepath = os.path.dirname(os.path.abspath(sys.argv[0]))  + r"/test.obj"
else:
    filepath = os.path.dirname(os.path.abspath(sys.argv[0]))  + r"\test.obj"
intersection: list = []
with open(filepath,'r') as fin:
    lines = fin.readlines()
    fin.close()

#Collect vertices
print ("Collect Verts")
Vertices: list = []
for line in lines:
    #Record each vertex
    if line[:2] == 'v ':
        newVector = [0, 0, 0]
        newVector[0] = (float)(line.split()[1])
        newVector[1] = (float)(line.split()[2])
        newVector[2] = (float)(line.split()[3])
        Vertices.append(newVector)

#Build faces
print ("Build Faces")
Faces: list = []
for line in lines:
    #Record each face with its coordinates
    if line[:2] == 'f ':
        newFace = []
        #re.sub('"', '', filepath)
        newFace.append(Vertices[(int)(re.sub('/', ' ',line.split()[1]).split()[0]) - 1])
        newFace.append(Vertices[(int)(re.sub('/', ' ',line.split()[2]).split()[0]) - 1])
        newFace.append(Vertices[(int)(re.sub('/', ' ',line.split()[3]).split()[0]) - 1])
        Faces.append(newFace)
print("\n FACES:")

"""while(True):
    for face in Faces:
        start_time = time.time_ns()
        intersection.append(isect_tri_plane_v3(face[0], face[1], face[2], p_co, p_no))
        finalIntersection =list(filter(lambda x: len(x) is not 0, intersection))
        print("--- %s seconds ---" % (time.time_ns() - start_time))
        #if is_point_on_line_v3(newPoint, p0, p1):
"""

#else:
#    print(False)
iters: int = 0
while iters < 500:
    start_time = time.time_ns()
    #SLICING
    slices = []
    xpix = []
    ypix = []
    for face in Faces:
        slices.append(isect_tri_plane_v3(face[0], face[1], face[2], p_co, p_no))
    finalIntersection =list(filter(lambda x: len(x) == 2, slices))


    #RASTERIZATION
    planarOffset = dir_to_euler(p_no)[1]

    for inter in finalIntersection:
        #Align lines to the z-plane
        inter[0] = align_vector(inter[0], planarOffset)
        inter[1] = align_vector(inter[1], planarOffset)

        """#Convert lines to pixels
        linepix = ski.draw.line(int(inter[0][0]), int(inter[0][1]), int(inter[1][0]), int(inter[1][1]))
        for linexpix in linepix[0]:
            xpix.append(linexpix)
        for lineypix in linepix[1]:
            ypix.append(lineypix)"""

    
    #plt.scatter(xpix, ypix)
    #plt.show()

    p_no = align_vector(p_no, 10)
    
    if p_no[0] < 0.001 and p_no[0] > -0.001 :
        p_no[0] = 0
    if p_no[2] < 0.001 and p_no[2] > -0.001 :
        p_no[2] = 0

    iters += 1
    print("--- %s seconds ---" % (time.time_ns() - start_time))



#line = ski.draw.line(0, 0, -5, -8)

#for point in :
#    xpix.append(point[0])
#    ypix.append(point[1])
#plt.scatter(xpix, ypix)
#plt.ylabel('some numbers')
#plt.show()


