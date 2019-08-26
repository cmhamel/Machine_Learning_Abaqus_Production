import numpy as np
import random

from abaqus import *
from abaqusConstants import *
import sketch
import part
import assembly
import material
import interaction
from mesh import *
import time
import visualization
import odbAccess


#
# units are N and mm
#


def read_genome(f):

    with open(f, 'r') as gene:
        gene.readline()
        gene.readline()
        voxel_sizes = gene.readline().split(', ')
        nx = int(voxel_sizes[0])
        ny = int(voxel_sizes[1])
        raw_locations = gene.readline()
        print(raw_locations)

    locations = np.zeros((nx, ny), dtype=int)
    n = 0

    while n < len(raw_locations.split()):
        for i in range(nx):
            for j in range(ny):
                locations[i][j] = int(raw_locations.split()[n])
                n = n + 1
    return nx, ny, locations


class ActiveComposite:

    def __init__(self, length, height,
                 voxel_size_x, voxel_size_y,
                 locations):
        self.length = length
        self.height = height
        self.voxel_size_x = voxel_size_x
        self.voxel_size_y = voxel_size_y
        self.locations = locations
        self.model_name = 'Candidate' + str(random.randint(0, 100000))
        self.m = mdb.Model(name=self.model_name)

    def generate_geometry(self):

        s = self.m.ConstrainedSketch(name='Base-Geometry',
                                     sheetSize=10)

        s.rectangle(point1=(0.0, 0.0),
                    point2=(self.length, self.height))

        part = self.m.Part(name='Base-Geometry',
                           dimensionality=TWO_D_PLANAR,
                           type=DEFORMABLE_BODY)

        part.BaseShell(sketch=s)

        part.Set(name='Strip',
                 faces=part.faces)

    def voxelize_faster(self):

        part = self.m.parts['Base-Geometry']

        n = 0
        m = 0

        temp_face = self.m.ConstrainedSketch(name='Partition-X',
                                             sheetSize=10)

        while n < self.length / self.voxel_size_x:
            x1 = 0.0 + n * self.voxel_size_x
            y1 = 0.0
            x2 = x1
            y2 = self.height

            temp_face.Line(point1=(x1, y1),
                           point2=(x2, y2))

            n = n + 1

        while m < self.height / self.voxel_size_y:
            x1 = 0.0
            y1 = 0.0 + m * self.voxel_size_y
            x2 = self.length
            y2 = y1

            temp_face.Line(point1=(x1, y1),
                           point2=(x2, y2))

            m = m + 1

        main_face = part.sets['Strip'].faces[0]

        part.PartitionFaceBySketch(faces=main_face,
                                   sketch=temp_face)

    def voxelize_geometry(self):
        part = self.m.parts['Base-Geometry']

        n = 0

        temp_face = self.m.ConstrainedSketch(name='Partition-X',
                                             sheetSize=10)

        while n < self.length/self.voxel_size_x:

            x1 = 0.0 + n*self.voxel_size_x
            y1 = 0.0
            x2 = x1 + self.voxel_size_x
            y2 = self.height

            temp_face.rectangle(point1=(x1, y1),
                                point2=(x2, y2))

            n = n + 1

        main_face = part.sets['Strip'].faces[0]

        part.PartitionFaceBySketch(faces=main_face,
                                   sketch=temp_face)

        for i in range(n):

            temp_face = self.m.ConstrainedSketch(name='Partition-Y',
                                                 sheetSize=10)
            m = 0

            while m < self.height/self.voxel_size_y:

                x1 = 0.0
                y1 = 0.0 + m*self.voxel_size_y
                x2 = self.length
                y2 = y1 + self.voxel_size_y

                temp_face.rectangle(point1=(x1, y1),
                                    point2=(x2, y2))

                m = m + 1

            main_face = part.sets['Strip'].faces[-1-i]

            part.PartitionFaceBySketch(faces=main_face,
                                       sketch=temp_face)

    def generate_assembly(self):
        a = self.m.rootAssembly

        a.Instance(name='Structure',
                   part=self.m.parts['Base-Geometry'],
                   dependent=ON)

    def create_sections_and_assign_materials(self):

        p = self.m.parts['Base-Geometry']
        a = self.m.rootAssembly

        passive_mat = self.m.Material(name='Passive_mat')
        passive_mat.Hyperelastic(type=ARRUDA_BOYCE,
                                 testData=OFF,
                                 table=((.5, 5., 2.e-3,),))
        #passive_mat.Expansion(table=((0.0017,),))

        active_mat = self.m.Material(name='Active_mat')
        active_mat.Hyperelastic(type=ARRUDA_BOYCE,
                                testData=OFF,
                                table=((5., 5., 2.e-3,),))
        active_mat.Expansion(table=((0.0005,),))

        n = int(self.length / self.voxel_size_x)
        m = int(self.height / self.voxel_size_y)

        for i in range(n):
            for j in range(m):
                x1 = 0.0 + i*self.voxel_size_x
                y1 = 0.0 + j*self.voxel_size_y
                x2 = x1 + self.voxel_size_x
                y2 = y1 + self.voxel_size_y

                f_temp = p.faces.getByBoundingBox(xMin=x1,
                                                  xMax=x2,
                                                  yMin=y1,
                                                  yMax=y2,
                                                  zMin=-.1,
                                                  zMax=.1)

                p.Set(name='Section-' + str(i) + '-' + str(j),
                      faces=f_temp)

        boo = True

        active = ()
        passive = ()

        for i in range(n):
            for j in range(m):
                s = p.sets['Section-' + str(i) + '-' + str(j)]

                if self.locations[i][j] == 1:
                    if boo:
                        active = active + (s.faces,)

                        boo = False
                    else:
                        active = active + (s.faces,)
                else:
                    passive = passive + (s.faces,)

        p.Set(name='Active-Set',
              faces=active)

        p.Set(name='Passive-Set',
              faces=passive)

        p.Set(name='All',
              faces=p.faces)

        self.m.HomogeneousSolidSection(material='Passive_mat',
                                       name='Passive')
        self.m.HomogeneousSolidSection(material='Active_mat',
                                       name='Active')

        p.SectionAssignment(region=p.sets['Passive-Set'],
                            sectionName='Passive')
        p.SectionAssignment(region=p.sets['Active-Set'],
                            sectionName='Active')

    def assign_boundary_conditions(self, T_Cold, T_Hot):

        a = self.m.rootAssembly

        p = self.m.parts['Base-Geometry']

        inst = a.instances['Structure']

        left_edge = inst.edges.getByBoundingBox(xMin=-0.001,
                                                yMin=-0.001,
                                                zMin=-0.001,
                                                xMax=0.001,
                                                yMax=self.height + 0.001,
                                                zMax=0.001)

        a.Set(edges=left_edge,
              name='Left_edge')

        top_edge = inst.edges.getByBoundingBox(xMin=-0.001,
                                               yMin=self.height - 0.001,
                                               zMin=-0.001,
                                               xMax=self.length + .001,
                                               yMax=self.height + 0.001,
                                               zMax=0.001)

        a.Set(edges=top_edge,
              name='Top_edge')

        vertex = inst.vertices.getByBoundingBox(xMin=self.length - 0.001,
                                                yMin=self.height - 0.001,
                                                zMin=-0.001,
                                                xMax=self.length + 0.001,
                                                yMax=self.height + 0.001,
                                                zMax=0.001)

        a.Set(vertices=vertex,
              name='Top_Vertex')

        self.m.Temperature(name='Initial_T_Cold_Active',
                           createStepName='Initial',
                           region=a.instances['Structure'].sets['All'],
                           magnitudes=(((T_Cold,))))

        self.m.DisplacementBC(name='LeftEdgeFixed',
                              createStepName='Initial',
                              region=a.sets['Left_edge'],
                              u1=0,
                              u2=0)

        self.m.StaticStep(name='Heating',
                          previous='Initial',
                          initialInc=0.1,
                          maxInc=0.1,
                          minInc=0.0001,
                          nlgeom=ON)

        self.m.Temperature(name='T_Hot_Active',
                           createStepName='Heating',
                           region=a.instances['Structure'].sets['All'],
                           magnitudes=(((T_Hot,))))

        # self.m.ConcentratedForce(name='End',
        #                          createStepName='Heating',
        #                          region=a.sets['Top_Vertex'],
        #                          cf1=0.0,
        #                          cf2=-0.001)

    def mesh_structure(self):

        global_seed = min(self.voxel_size_x / 3.0, self.voxel_size_y / 3.0)

        p = self.m.parts['Base-Geometry']

        p.setMeshControls(regions=p.faces,
                          elemShape=QUAD)

        p.setElementType(elemTypes=(ElemType(elemCode=CPE4H,
                                             elemLibrary=STANDARD),),
                         regions=p.sets['All'])

        p.seedPart(size=global_seed)

        p.generateMesh()

    def run_simulation(self, job_name):

        mdb.Job(contactPrint=OFF, description='', echoPrint=OFF,
                explicitPrecision=SINGLE, historyPrint=OFF,
                memory=90, memoryUnits=PERCENTAGE, model=self.model_name,
                modelPrint=OFF, multiprocessingMode=DEFAULT, name=job_name,
                nodalOutputPrecision=SINGLE, numCpus=1, numDomains=1,
                parallelizationMethodExplicit=DOMAIN, scratch='', type=ANALYSIS,
                userSubroutine='')
        mdb.jobs[job_name].submit()
        mdb.jobs[job_name].waitForCompletion()

        if mdb.jobs[job_name].status == ABORTED or mdb.jobs[job_name].status == TERMINATED:
            return True
        else:
            return False

    @staticmethod
    def post_process(aborted, job_name):
        x = []
        y = []

        out = 'output-' + job_name.split('.')[0] + '.txt'

        fout = open(out, 'w+')
        fout.write('*Initial Position\n')

        if aborted:
            fout.write('*Simulation Failed\n')

        try:
            odb = session.openOdb(name=job_name + '.odb')
        except RunTimeError:
            pass

        if len(odb.steps['Heating'].frames) == 0:
            fout.write('*Simulation Failed\n')

        frame = odb.steps['Heating'].frames[-1]

        nodes = odb.rootAssembly.nodeSets['TOP_EDGE']

        U = frame.fieldOutputs['U'].getSubset(region=nodes)

        for node in nodes.nodes[0]:
            fout.write(str(node.label) + ',' + str(node.coordinates[0]) + ',' + str(node.coordinates[1]) + '\n')

        fout.write('*Displacements\n')

        for v in U.values:
            fout.write(str(v.nodeLabel) + ',' + str(v.data[0]) + ',' + str(v.data[1]) + '\n')

        fout.close()
        odb.close


def main(gene_file, job_name):

    T_COLD = 20.0
    T_HOT = 120.0

    length = 50.0
    height = 2.5
    # voxel_size_x = length / 100.0
    # voxel_size_y = voxel_size_x

    nx, ny, locations = read_genome(gene_file)

    voxel_size_x = length / nx
    voxel_size_y = height / ny

    model1 = ActiveComposite(length, height,
                             voxel_size_x, voxel_size_y,
                             locations)

    model1.generate_geometry()
    model1.voxelize_faster()
    model1.generate_assembly()
    model1.create_sections_and_assign_materials()
    model1.assign_boundary_conditions(T_COLD, T_HOT)
    model1.mesh_structure()
    aborted = model1.run_simulation(job_name)
    model1.post_process(aborted, job_name)


if __name__ == '__main__':

    args = sys.argv

    #for GUI
    #my_args = args[8:]

    #print(my_args)
    # for noGUI
    my_args = args[8:]

    gene_file_in = my_args[0]
    job_name_in = my_args[1]
    #gene_file = ''

    main(gene_file_in, job_name_in)
