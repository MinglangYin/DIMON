import vtk
import numpy as np
import meshio
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util import numpy_support #import vtk_to_numpy


def convert_uvc(coords, center):
    # AB
    z_min, z_max = coords[:, 2:3].min(), coords[:, 2:3].max()
    AB = (coords[:, 2:3]-z_min)/(z_max-z_min)

    # TM
    r = np.sqrt( np.sum((coords - center)**2, axis=1) )
    r_min, r_max = r.min(), r.max()
    TM = (r-r_min)/(r_max-r_min)
    TM = TM[:, None]

    # PHI    
    PHI = np.arctan2(coords[:, 1:2] - center[1], coords[:, 0:1] - center[0])
    PHI_new = (2*np.pi + PHI)*(PHI < 0) + PHI*(PHI > 0)
    PHI = (PHI_new - PHI_new.min())/(PHI_new.max() - PHI_new.min())*2*np.pi

    return AB, TM, PHI

def main():
    ## add field to vtk
    fileName = "foo.vtk"
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(fileName)
    reader.Update()
    mesh = reader.GetOutput()

    ## get vtk coordinate
    Point_cordinates = reader.GetOutput().GetPoints().GetData()
    numpy_coordinates = numpy_support.vtk_to_numpy(Point_cordinates)

    # convert xyz to uvc
    center = [0.5, 0.5, 0.5] # assume a semi-spherical shell
    AB, TM, PHI = convert_uvc(numpy_coordinates, center)
    uvc = np.concatenate((AB, TM, PHI), axis=1)
    # np.savetxt("canonical_uvc.txt", uvc)

    # load canonical stim
    stims = np.loadtxt("LATs_train_pred.txt")
    LATs_true = np.loadtxt("LATs_true_case0.txt")
    # stims = np.loadtxt("LATs_test_pred.txt")
    # LATs_true = np.loadtxt("LATs_true_case70.txt")

    # Add data set and write VTK file
    meshNew = dsa.WrapDataObject(mesh)

    meshNew.PointData.append(stims[:, 0], "stim0_pred")
    meshNew.PointData.append(stims[:, 1], "stim1_pred")
    meshNew.PointData.append(stims[:, 2], "stim2_pred")
    meshNew.PointData.append(stims[:, 3], "stim3_pred")
    meshNew.PointData.append(stims[:, 4], "stim4_pred")
    meshNew.PointData.append(stims[:, 5], "stim5_pred")
    meshNew.PointData.append(stims[:, 6], "stim6_pred")

    meshNew.PointData.append(LATs_true[:, 0], "stim0_true")
    meshNew.PointData.append(LATs_true[:, 1], "stim1_true")
    meshNew.PointData.append(LATs_true[:, 2], "stim2_true")
    meshNew.PointData.append(LATs_true[:, 3], "stim3_true")
    meshNew.PointData.append(LATs_true[:, 4], "stim4_true")
    meshNew.PointData.append(LATs_true[:, 5], "stim5_true")
    meshNew.PointData.append(LATs_true[:, 6], "stim6_true")

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName("vis_LATs_case0.vtk")
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()
    

if __name__ == "__main__":
    main()
