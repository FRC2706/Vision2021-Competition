#
import numpy as np
import cv2

black = (0, 0, 0)
purple = (165, 0, 120)
green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)

booWrite = False

def outer_bottoms(apprx, left_side):
    """
    """
    # can't find sort for approx array output, so re-create as simple list
    # pull each row and remove layer of []
    [row0] = apprx[0]
    [row1] = apprx[1]
    [row2] = apprx[2]
    [row3] = apprx[3]

    # create normal list and append points into it
    lst = []
    lst.append(row0)
    lst.append(row1)
    lst.append(row2)
    lst.append(row3)
    
    # sort list by second column or y axis ascending
    lst.sort(key=lambda elem: elem[1])
    # keep only bottom two in new list
    ylst = lst[-2:]

    # sort list by first column or x axis
    ylst.sort(key=lambda elem: elem[0])

    # create list to return with outside corners first, inside second
    xlst = []
    if left_side:
        xlst.append(ylst[0])
        xlst.append(ylst[1])
    else:
        xlst.append(ylst[1])
        xlst.append(ylst[0])

    # return list in desired form
    return xlst

def is_cv3():
    # if we are using OpenCV 3.X, then our cv2.__version__ will start
    # with '3.'
    return check_opencv_version("3.")

def check_opencv_version(major, lib=None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        import cv2 as lib
    # return whether or not the current OpenCV version matches the
    # major version number
    return lib.__version__.startswith(major) 

def get_four(boundr, width, height, contour):
    """
    """
    # carry coords of bounding box into function 
    (xb,yb,wb,hb) = boundr

    # ignore small contours in case caller does not have filter
    if (width > 10) and (height > 10):

        #calculate middle, thickness and radius for circles
        middle = int(width/2)
        thickness = int(height/10)
        divider = int(height/10)

        # make mask from blank image and contour
        binary_mask = np.zeros([height,width,1],dtype=np.uint8)
        cv2.drawContours(binary_mask, [contour], -1, 255, cv2.FILLED)
        if booWrite: cv2.imwrite('./01-binary_mask.jpg', binary_mask)

        #find leftmost and rightmost
        leftmost = tuple(contour[contour[:,:,0].argmin()][0])
        rightmost = tuple(contour[contour[:,:,0].argmax()][0])
        
        # make negative of mask and close with line, divide in half
        negative_mask = cv2.bitwise_not(binary_mask)
        if booWrite: cv2.imwrite('02-negative_mask.jpg', negative_mask)
        modified_mask = negative_mask.copy()
        cv2.line(modified_mask, leftmost, rightmost, black, thickness, cv2.LINE_AA)
        cv2.line(modified_mask, (middle,0), (middle,height), black, divider, cv2.LINE_AA)
        if booWrite: cv2.imwrite('03-added_lines.jpg', modified_mask)

        #if booWrite: img123visual4 = np.hstack([binary_mask, negative_mask, modified_mask])

        # make hull and image of hull for bitwise_and
        hull = cv2.convexHull(contour)
        hull_mask = np.zeros([height,width,1],dtype=np.uint8)    
        cv2.drawContours(hull_mask, [hull], -1, 255, cv2.FILLED)
        if booWrite: cv2.imwrite('04-binary_hull.jpg', hull_mask)

        # bitwise_and the modified negative_mask and hull_mask
        bitwise_bottoms = cv2.bitwise_and(modified_mask,hull_mask)
        if booWrite: cv2.imwrite('05-bitwise_bottoms.jpg',bitwise_bottoms)

        # make contours of bitwise_and and keep largest two
        if is_cv3():
            imgFindContourReturn, bitwiseContours, hierarchy = cv2.findContours(bitwise_bottoms, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            bitwiseContours, hierarchy = cv2.findContours(bitwise_bottoms, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sortedContours = sorted(bitwiseContours, key = cv2.contourArea, reverse = True)[:2]

        # sort contours so we know left vs right
        # https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
        boundingBoxes = [cv2.boundingRect(c) for c in sortedContours]
        (sortedContours, boundingBoxes) = zip(*sorted(zip(sortedContours, boundingBoxes),
            key=lambda b:b[1][0], reverse=False))

        # approximate four corners of both, 0.035 by trial and error to deliver 4 points
        leftHalf = sortedContours[0]
        epsilon = 0.035*cv2.arcLength(leftHalf,True)
        leftApprox = cv2.approxPolyDP(leftHalf,epsilon,True)
        rightHalf = sortedContours[1]
        epsilon = 0.035*cv2.arcLength(rightHalf,True)
        rightApprox = cv2.approxPolyDP(rightHalf,epsilon,True)

        # draw approximated contours on colourized mask
        two_contours = cv2.cvtColor(bitwise_bottoms,cv2.COLOR_GRAY2RGB)
        cv2.drawContours(two_contours, [leftApprox], -1, purple, 5)
        cv2.drawContours(two_contours, [rightApprox], -1, purple, 5)
        if booWrite: cv2.imwrite('06-two_contours.jpg', two_contours)

        #if booWrite: img456visual4 = np.hstack([hull_mask, bitwise_bottoms, two_bottoms])

        # draw points of approx on top of contours 
        approx_points = cv2.cvtColor(bitwise_bottoms,cv2.COLOR_GRAY2RGB)
        cv2.drawContours(approx_points, leftApprox, -1, green, thickness)
        cv2.drawContours(approx_points, rightApprox, -1, red, thickness)
        if booWrite: cv2.imwrite('07-approx_points.jpg',approx_points)

        if len(leftApprox) is 4 and len(rightApprox) is 4:
            # seems it is dificult to sort output of approx function, use custom sort
            leftBottoms = outer_bottoms(leftApprox, True)
            rightBottoms = outer_bottoms(rightApprox, False)

            # first entries are outer points
            arrayLeft = leftBottoms[0]
            arrayRight = rightBottoms[0]

            # extract out desired bottom corners
            [blx, bly] = arrayLeft
            [brx, bry] = arrayRight
            bottomleft = (blx, bly)
            bottomright = (brx, bry)

            # potential fifth point, average of inner points, from split of hull
            [lx5, ly5] = leftBottoms[1]
            [rx5, ry5] = rightBottoms[1]
            fifth = (int((lx5+rx5)/2),int((ly5+ry5)/2))
            bottomcenter = fifth

            # draw found corners
            color_bottoms = cv2.cvtColor(bitwise_bottoms, cv2.COLOR_GRAY2RGB)
            cv2.circle(color_bottoms, bottomleft, thickness, green, -1)
            cv2.circle(color_bottoms, bottomright, thickness, red, -1)
            cv2.circle(color_bottoms, bottomcenter, thickness, blue, -1)
            if booWrite: cv2.imwrite('08-bottom_points.jpg',color_bottoms)

            # draw four corners and center
            four_corners = cv2.cvtColor(bitwise_bottoms, cv2.COLOR_GRAY2RGB)
            cv2.circle(four_corners, leftmost, int(thickness/2), green, -1)
            cv2.circle(four_corners, rightmost, int(thickness/2), red, -1)
            cv2.circle(four_corners, bottomleft, int(thickness/2), blue, -1)
            cv2.circle(four_corners, bottomcenter, int(thickness/2), blue, -1)
            cv2.circle(four_corners, bottomright, int(thickness/2), blue, -1)
            if booWrite: cv2.imwrite('09-four_points.jpg',four_corners)

            if booWrite: img789visual4 = np.hstack([approx_points, color_bottoms, four_corners])

            if booWrite: 
                imgTraceVisual4 = np.vstack([img789visual4])
                cv2.imshow('Visual4 Trace Steps', imgTraceVisual4)
                #cv2.imwrite('10-Vision4Steps',imgTraceVisual4)
                cv2.moveWindow('Visual4 Trace Steps',350,760)

            rul, rbl, rbc, rbr, rur = leftmost, bottomleft, bottomcenter, bottomright, rightmost

            rulx, ruly = rul
            ulx = rulx + xb
            uly = ruly + yb

            rblx, rbly = rbl
            blx = rblx + xb
            bly = rbly + yb

            rbcx, rbcy = rbc
            bcx = rbcx + xb
            bcy = rbcy + yb

            rbrx, rbry = rbr
            brx = rbrx + xb
            bry = rbry + yb

            rurx, rury = rur
            urx = rurx + xb
            ury = rury + yb

            return True, [(float(ulx),float(uly)), (float(blx),float(bly)), (float(bcx),float(bcy)), (float(brx),float(bry)), (float(urx),float(ury))]

        else:
            return False, None
    else:
        return False, None
