import numpy as np
import random
import csv
import sys

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
                 nx, ny,
                 voxel_size_x, voxel_size_y,
                 locations):
        self.length = length
        self.height = height
        self.nx = nx
        self.ny = ny
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

    def voxelization(self):
        part = self.m.parts['Base-Geometry']

        for i in range(self.nx):

            temp_name = 'Partition-X-' + str(i)

            temp_sketch = self.m.ConstrainedSketch(name=temp_name,
                                                   sheetSize=10)
            for j in range(self.ny):
                x1 = 0.0 + i * self.voxel_size_x
                y1 = 0.0 + j * self.voxel_size_y
                x2 = 0.0 + (i + 1) * self.voxel_size_x
                y2 = 0.0 + (j + 1) * self.voxel_size_y

                temp_sketch.rectangle(point1=(x1, y1),
                                      point2=(x2, y2))

            main_face = part.faces.findAt(coordinates=((self.length, self.height, 0.0),))

            part.PartitionFaceBySketch(faces=main_face,
                                       sketch=temp_sketch)

    def generate_assembly(self):
        a = self.m.rootAssembly

        a.Instance(name='Structure',
                   part=self.m.parts['Base-Geometry'],
                   dependent=ON)

    def create_sections_and_assign_materials(self):

        p = self.m.parts['Base-Geometry']

        passive_mat = self.m.Material(name='Passive_mat')
        passive_mat.Hyperelastic(type=NEO_HOOKE,
                                 testData=OFF,
                                 table=((.1, 2.e-3,),))
        passive_mat.Viscoelastic(domain=TIME,
                                 time=PRONY,
                                 table=((0.0957966764418377, 0, 0.00000804,),
                                        (0.0987292277614858, 0, 0.0000928,),
                                        (0.117302052785924, 0, 0.000827,),
                                        (0.131964809384164, 0, 0.006032,),
                                        (0.142717497556207, 0, 0.036577,),
                                        (0.162267839687195, 0, 0.187428,),
                                        (0.148093841642229, 0, 0.860502,),
                                        (0.0689149560117302, 0, 4.728029,),
                                        (0.00664438416422287, 0, 20,),
                                        (0.00024714926686217, 0, 83.13049,),
                                        (0.000566467253176931, 0, 375.1723,),
                                        (0.00168797360703812, 0, 1848.755,),
                                        (2.22441300097752E-05, 0, 10000,),
                                        (0.0248336500488759, 0, 67650.29,),
                                        (0.000100122727272727, 0, 935839.6,),))

        passive_mat.viscoelastic.Trs(definition=USER)

        active_mat_1 = self.m.Material(name='Active_mat_1')
        active_mat_1.Hyperelastic(type=NEO_HOOKE,
                                testData=OFF,
                                table=((1.25, 2.e-3,),))
        active_mat_1.Viscoelastic(domain=TIME,
                                  time=PRONY,
                                  table=((0.0453333333333333, 0, 0.0000001,),
                                         (0.0501333333333333, 0, 0.000000993,),
                                         (0.0565333333333333, 0, 0.00001,),
                                         (0.0637333333333333, 0, 0.0000908,),
                                         (0.0714666666666667, 0, 0.00074,),
                                         (0.0781333333333333, 0, 0.005374,),
                                         (0.0821333333333333, 0, 0.035368,),
                                         (0.0776, 0, 0.2,),
                                         (0.076, 0, 0.954957,),
                                         (0.0368, 0, 3.182197,),
                                         (0.0432, 0, 7.497457,),
                                         (0.0474666666666667, 0, 25.11365,),
                                         (0.0408, 0, 87.11596,),
                                         (0.0354666666666667, 0, 283.7953,),
                                         (0.0325333333333333, 0, 905.6253,),
                                         (0.0298666666666667, 0, 3025.975,),
                                         (0.0261588098666667, 0, 10000,),
                                         (0.0222029186666667, 0, 32677.22,),
                                         (0.0175212149333333, 0, 96510.16,),
                                         (0.0157653898666667, 0, 267333.4,),
                                         (0.0138459149333333, 0, 773277.7,),
                                         (0.0119384872, 0, 2339554,),
                                         (0.00922653146666667, 0, 7613180,),
                                         (0.0057938992, 0, 26070126,),
                                         (0.00266540773333333, 0, 100000000,),
                                         (0.000777802133333333, 0, 522000000,),
                                         (0.00025523672, 0, 5770000000,),))

        active_mat_1.viscoelastic.Trs(definition=USER)

        active_mat_2 = self.m.Material(name='Active_mat_2')
        active_mat_2.Hyperelastic(type=NEO_HOOKE,
                                  testData=OFF,
                                  table=((0.55, 2.e-3,),))
        active_mat_2.Viscoelastic(domain=TIME,
                                  time=PRONY,
                                  table=((0.100055030266647, 0, 0.0001),
                                         (0.0917171110777594, 0, 0.000657),
                                         (0.0987209631964247, 0, 0.003872),
                                         (0.101722614104424, 0, 0.02),
                                         (0.116730868644421, 0, 0.1),
                                         (0.126069338135975, 0, 0.576863),
                                         (0.0973868961262027, 0, 3.401616),
                                         (0.0717061050244301, 0, 20),
                                         (0.0490269648306569, 0, 96.82391),
                                         (0.0317552877415912, 0, 362.9461),
                                         (0.0210541297713743, 0, 1000),
                                         (0.0207087564827322, 0, 2671.527),
                                         (0.0173759921290043, 0, 7912.87),
                                         (0.0141326793069522, 0, 23498.79),
                                         (0.0117416075508196, 0, 71461.38),
                                         (0.00930430136575116, 0, 228551.6),
                                         (0.0069240645688462, 0, 726401),
                                         (0.00518032551236513, 0, 2277776),
                                         (0.00376269548251538, 0, 7091525),
                                         (0.00277012056631147, 0, 21997171),))

        active_mat_2.viscoelastic.Trs(definition=USER)

        for i in range(self.nx):
            for j in range(self.ny):
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

        boo_1 = True
        boo_2 = True

        active_1 = ()
        active_2 = ()
        passive = ()

        for i in range(self.nx):
            for j in range(self.ny):
                s = p.sets['Section-' + str(i) + '-' + str(j)]

                if self.locations[i][j] == 1:
                    if boo_1:
                        active_1 = active_1 + (s.faces,)

                        boo_1 = False
                    else:
                        active_1 = active_1 + (s.faces,)
                elif self.locations[i][j] == 2:
                    if boo_2:
                        active_2 = active_2 + (s.faces,)

                        boo_2 = False
                    else:
                        active_2 = active_2 + (s.faces,)
                else:
                    passive = passive + (s.faces,)

        active_set_1 = p.Set(name='Active-Set-1',
                             faces=active_1)

        active_set_2 = p.Set(name='Active-Set-2',
                             faces=active_2)

        passive_set = p.Set(name='Passive-Set',
                            faces=passive)

        all_set = p.Set(name='All',
                        faces=p.faces)

        self.m.HomogeneousSolidSection(material='Passive_mat',
                                       name='Passive')
        self.m.HomogeneousSolidSection(material='Active_mat_1',
                                       name='Active_1')
        self.m.HomogeneousSolidSection(material='Active_mat_2',
                                       name='Active_2')

        p.SectionAssignment(region=p.sets['Passive-Set'],
                            sectionName='Passive')
        p.SectionAssignment(region=p.sets['Active-Set-1'],
                            sectionName='Active_1')
        p.SectionAssignment(region=p.sets['Active-Set-2'],
                            sectionName='Active_2')

    def create_steps_and_assign_boundary_conditions(self):

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

        right_edge = inst.edges.getByBoundingBox(xMin=self.length-0.001,
                                                 yMin=- 0.001,
                                                 zMin=-0.001,
                                                 xMax=self.length + .001,
                                                 yMax=self.height + 0.001,
                                                 zMax=0.001)

        a.Set(edges=right_edge,
              name='Right_edge')

        vertex = inst.vertices.getByBoundingBox(xMin=self.length - 0.001,
                                                yMin=self.height - 0.001,
                                                zMin=-0.001,
                                                xMax=self.length + 0.001,
                                                yMax=self.height + 0.001,
                                                zMax=0.001)

        a.Set(vertices=vertex,
              name='Top_Vertex')

        vertex = inst.vertices.getByBoundingBox(xMin=-0.001,
                                                yMin=-0.001,
                                                zMin=-0.001,
                                                xMax=0.001,
                                                yMax=0.001,
                                                zMax=0.001)

        a.Set(vertices=vertex,
              name='Pinned_Vertex')
        #
        # steps
        #
        self.m.ViscoStep(name='Loading',
                         previous='Initial',
                         timePeriod=60,
                         initialInc=1,
                         minInc=1.e-8,
                         maxInc=25.0,
                         nlgeom=ON,
                         cetol=0.05,
                         maxNumInc=10000)

        self.m.ViscoStep(name='Hold',
                         previous='Loading',
                         timePeriod=60,
                         initialInc=1,
                         nlgeom=ON,
                         cetol=0.05,
                         maxNumInc=10000)

        self.m.ViscoStep(name='Release',
                         previous='Hold',
                         timePeriod=60,
                         initialInc=1,
                         minInc=1.e-8,
                         maxInc=5.0,
                         nlgeom=ON,
                         cetol=0.05,
                         maxNumInc=10000)

        # self.m.ViscoStep(name='Heat',
        #                  previous='Release',
        #                  timePeriod=60,
        #                  initialInc=1,
        #                  minInc=1.e-8,
        #                  maxInc=5.0,
        #                  nlgeom=ON,
        #                  cetol=0.05)
        #
        # initial boundary conditions
        #
        self.m.DisplacementBC(name='LeftEdgeFixed',
                              createStepName='Initial',
                              region=a.sets['Left_edge'],
                              u1=0)

        self.m.DisplacementBC(name='PinnedNode',
                              createStepName='Initial',
                              region=a.sets['Pinned_Vertex'],
                              u1=0,
                              u2=0)
        #
        # non-initial boundary conditions
        #
        load_bc = self.m.DisplacementBC(name='RightEdgeLoad',
                                        createStepName='Loading',
                                        region=a.sets['Right_edge'],
                                        u1=7.0,
                                        u2=0.0)

        load_bc.deactivate(stepName='Release')
        #
        # thermal history below
        #
        temp_field = self.m.Temperature(name='Initial-Temp',
                                        createStepName='Initial',
                                        region=inst.sets['All'],
                                        magnitudes=70)
        #
        # amplitude for cooling
        #
        a_cool = self.m.EquallySpacedAmplitude(name='cooling',
                                               fixedInterval=60,
                                               data=(4, 1,))
        #
        temp_field.setValuesInStep(stepName='Hold',
                                   amplitude='cooling',
                                   magnitudes=0.0)

        # a_heat = self.m.EquallySpacedAmplitude(name='heating',
        #                                        fixedInterval=60,
        #                                        data=(.25, 1,))
        #
        # temp_field.setValuesInStep(stepName='Heat',
        #                            amplitude='heating',
        #                            magnitudes=30)

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
                userSubroutine='utrs-std.obj')
        mdb.jobs[job_name].submit()
        mdb.jobs[job_name].waitForCompletion()

        if mdb.jobs[job_name].status == ABORTED or \
                mdb.jobs[job_name].status == TERMINATED:
            return True
        else:
            return False

    @staticmethod
    def csv_output(aborted, job_name):
        out = 'output-' + job_name.split('.')[0] + '.csv'

        if aborted:
            # fout.write('*Simulation Failed\n')
            return

        try:
            odb = session.openOdb(name=job_name + '.odb')
        except RunTimeError:
            pass

        if len(odb.steps['Release'].frames) == 0:
            # fout.write('*Simulation Failed\n')
            return

        nodes = odb.rootAssembly.nodeSets['TOP_EDGE']
        node_labels, x_coords, y_coords = [], [], []
        for node in nodes.nodes[0]:
            node_labels.append(node.label)
            x_coords.append(node.coordinates[0])
            y_coords.append(node.coordinates[1])

        fieldnames = ['node_label',
                      'x_coordinate', 'y_coordinate',
                      'x_displacement', 'y_displacement']

        with open(out, 'wb') as csv_file:
            frame = odb.steps['Release'].frames[-1]
            u = frame.fieldOutputs['U'].getSubset(region=nodes)
            disp_node_labels, x_disp_temp, y_disp_temp = [], [], []
            for v in u.values:
                disp_node_labels.append(v.nodeLabel)
                x_disp_temp.append(v.data[0])
                y_disp_temp.append(v.data[1])

            for i in range(len(node_labels)):
                if node_labels[i] != disp_node_labels[i]:
                    print('major problem')
                    return

            x_disp = x_disp_temp
            y_disp = y_disp_temp

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(node_labels)):
                node_dict = {'node_label': node_labels[i],
                             'x_coordinate': x_coords[i],
                             'y_coordinate': y_coords[i],
                             'x_displacement': x_disp[i],
                             'y_displacement': y_disp[i]}

                writer.writerow(node_dict)


def main(gene_file, job_name):

    nx, ny, locations = read_genome(gene_file)

    length = 70.0
    height = 2.0
    voxel_size_x = length / nx
    voxel_size_y = height / ny

    model1 = ActiveComposite(length, height,
                             nx, ny,
                             voxel_size_x, voxel_size_y,
                             locations)

    model1.generate_geometry()
    model1.voxelization()
    model1.generate_assembly()
    model1.create_sections_and_assign_materials()
    model1.create_steps_and_assign_boundary_conditions()
    model1.mesh_structure()
    aborted = model1.run_simulation(job_name)
    # model1.post_process(aborted, job_name)
    model1.csv_output(aborted, job_name)


if __name__ == '__main__':

    args = sys.argv

    for arg in args:
        if '.txt' in arg:
            index = args.index(arg)

    my_args = args[index:]

    gene_file_in = my_args[0]
    job_name_in = my_args[1]

    main(gene_file_in, job_name_in)
