-This set of pictures was taken to help understand solvePNP output.
-These pictures were taken at distances of 4, 7 and 10 feet with 2 inch squares.
-The real world coordinates for each series ABCD, can be stated below
-The center of the target is assumed like the inifite recharge outer goal center
-The coordinates are to the centroid of the squares
-The height of the camera is 16 inches above floor
-The height of the target is 51 inches above floor
-The tilt of the camera is upwards, tan(18/42) or 26.18 degrees up from horizontal

-A -> an M as in Merge
real_world_coordinates = np.array([ 
    [-2.5, 0.0, 0.0], # Upper left point
    [2.5, 0.0, 0.0], # Upper right point
    [-5.5625, 6.125, 0.0], # Bottom left point
    [0.0, 6.125, 0.0], # Bottom center point
    [5.5625, 6.125, 0.0], # Bottom right point
    ])

-B -> the above target upside down
real_world_coordinates = np.array([ 
    [-5.5625, 0.0, 0.0], # Upper left point
    [0.0, 0.0, 0.0], # Upper center point
    [5.5625, 0.0, 0.0], # Upper right point
    [-2.5, 6.125, 0.0], # Bottom left point
    [2.5, 6.125, 0.0], # Bottom right point
    ])

-C -> a squished diamond defined by four squares
real_world_coordinates = np.array([ 
    [-5.5625, 3.0, 0.0], # Left most Point
    [5.5625, 3.0, 0.0], # Right most Point
    [0.0, 0.0, 0.0], # Top most point
    [0.0, 6.125, 0.0], # Bottom most Point
    ]) 

-D #
real_world_coordinates = np.array([ 
    [-5.5625, 0.0, 0.0], # Upper left point
    [0.0, 0.0, 0.0], # Upper center point
    [-2.5, 6.125, 0.0], # Bottom left point
    [2.5, 6.125, 0.0], # Bottom right point
    ])