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
# units are N and m and A
#
magFieldVecX = 0.0
magFieldVecY = 1.0
magFieldVecZ = 0.0
magFieldMag = 0.05
Gshear = 300.0e3
Kbulk = 100.0 * Gshear


class HardMagneticSoftMaterial:

    def __init__(self, f, job_name,
                 length, height, depth):
        #
        # constructor that initializes all necessary variables for construction
        # of the model
        #
        self.f = f
        self.job_name = job_name
        self.length = length
        self.height = height
        self.depth = depth

        self.model_name = 'Candidate' + str(random.randint(0, 100000))
        self.m = mdb.Model(name=self.model_name)
        self.m.setValues(noPartsInputFile=ON)

        nx, nz, locations = self.read_genome(self.f)

        self.nx = nx
        self.nz = nz
        self.locations = locations

        self.unit_size_x = self.length / self.nx
        self.unit_size_z = self.depth / self.nz

    @staticmethod
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

    def generate_geometry(self):

        s = self.m.ConstrainedSketch(name='Base-Geometry',
                                     sheetSize=0.01)

        s.rectangle(point1=(0.0, 0.0),
                    point2=(self.length, self.height))

        part = self.m.Part(name='Base-Geometry',
                           dimensionality=THREE_D,
                           type=DEFORMABLE_BODY)

        part.BaseSolidExtrude(sketch=s,
                              depth=self.depth)

        part.Set(name='Sheet',
                 cells=part.cells)

    def partition_geometry(self):

        part = self.m.parts['Base-Geometry']

        n, m = 0, 0

        while n < self.length / self.unit_size_x - 1:
            x1 = 0.0
            y1 = 0.0
            z1 = 0.0 + (n + 1) * self.unit_size_z
            x2 = self.length
            y2 = self.height
            z2 = z1
            x3 = self.length
            y3 = 0.0
            z3 = z1

            part.PartitionCellByPlaneThreePoints(cells=part.cells,
                                                 point1=(x1, y1, z1),
                                                 point2=(x2, y2, z2),
                                                 point3=(x3, y3, z3))

            n = n + 1

        while m < self.depth / self.unit_size_z - 1:
            x1 = 0.0 + (m + 1) * self.unit_size_x
            y1 = 0.0
            z1 = 0.0
            x2 = x1
            y2 = self.height
            z2 = self.depth
            x3 = x1
            y3 = 0.0
            z3 = self.depth

            part.PartitionCellByPlaneThreePoints(cells=part.cells,
                                                 point1=(x1, y1, z1),
                                                 point2=(x2, y2, z2),
                                                 point3=(x3, y3, z3))

            m = m + 1

    def create_assembly(self):
        a = self.m.rootAssembly

        a.Instance(name='Structure',
                   part=self.m.parts['Base-Geometry'],
                   dependent=ON)

    def create_material_section(self):
        p = self.m.parts['Base-Geometry']

        for m in range(self.nx):
            for n in range(self.nz):
                x_min = 0.0 + m * self.unit_size_x - 0.0001
                x_max = 0.0 + (m + 1) * self.unit_size_x
                y_min = -0.0001
                y_max = self.height + 0.0001
                z_min = 0.0 + n * self.unit_size_z - 0.0001
                z_max = 0.0 + (n + 1) * self.unit_size_z + 0.0001
                cell = p.cells.getByBoundingBox(xMin=x_min, xMax=x_max,
                                                yMin=y_min, yMax=y_max,
                                                zMin=z_min, zMax=z_max)

                set = p.Set(name='Cell-' + str(m) + '-' + str(n),
                            cells=cell)

                elements = set.elements

                p.Set(name='ElSet-' + str(m) + '-' + str(n),
                      elements=elements)

        top_face = p.faces.getByBoundingBox(xMin=-0.0001, xMax=self.length + 0.0001,
                                            yMin=self.height - 0.0001, yMax=self.height + 0.0001,
                                            zMin=-0.0001, zMax=self.depth + 0.0001)

        p.Set(name='Top-Face',
              faces=top_face)

    def assign_field_values(self):

        p = self.m.parts['Base-Geometry']
        a = self.m.rootAssembly

        self.m.Field(name='Predefined Field-1',
                     createStepName='Initial',
                     region=a.sets['Structure.NAll'],
                     distributionType=UNIFORM,
                     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
                     fieldVariableNum=1,
                     magnitudes=(magFieldVecX,))

        self.m.Field(name='Predefined Field-2',
                     createStepName='Initial',
                     region=a.sets['Structure.NAll'],
                     distributionType=UNIFORM,
                     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
                     fieldVariableNum=2,
                     magnitudes=(magFieldVecY,))

        self.m.Field(name='Predefined Field-3',
                     createStepName='Initial',
                     region=a.sets['Structure.NAll'],
                     distributionType=UNIFORM,
                     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
                     fieldVariableNum=3,
                     magnitudes=(magFieldVecZ,))

        self.m.Field(name='Predefined Field-4',
                     createStepName='Initial',
                     region=a.sets['Structure.NAll'],
                     distributionType=UNIFORM,
                     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
                     fieldVariableNum=4,
                     magnitudes=(magFieldMag,))

    def assign_boundary_conditions(self):
        p = self.m.parts['Base-Geometry']
        a = self.m.rootAssembly

        fixed_faces = p.faces.getByBoundingBox(xMin=-0.0001, xMax=0.0001,
                                               yMin=-0.0001, yMax=self.height + 0.0001,
                                               zMin=-0.0001, zMax=self.depth + 0.0001)

        p.Set(name='Fixed',
              faces=fixed_faces)

        self.m.DisplacementBC(name='Fixed',
                              createStepName='Initial',
                              region=a.sets['Structure.Fixed'],
                              u1=0.0, u2=0.0, u3=0.0)

    def mesh_structure(self):

        global_seed = min(self.unit_size_x / 2, self.unit_size_z / 2)

        p = self.m.parts['Base-Geometry']

        p.setMeshControls(regions=p.cells,
                          elemShape=HEX)

        p.setElementType(elemTypes=(ElemType(elemCode=C3D8H,
                                             elemLibrary=STANDARD),),
                         regions=p.sets['Sheet'])

        p.seedPart(size=global_seed)

        p.generateMesh()

        nodes = p.nodes

        elements = p.elements

        p.Set(name='NAll',
              nodes=nodes)

        p.Set(name='ElAll',
              elements=elements)

    def write_input_file(self):
        job = mdb.Job(name=self.job_name,
                      model=self.model_name)

        job.writeInput()

    def modify_input_file(self):
        fil = self.job_name + '.inp'

        element_lines = []

        with open(fil, 'r') as f:
            lines_in = f.readlines()

        os.system('del ' + self.job_name + '.inp')

        element_bool = False

        with open(fil, 'w') as f:
            for line in lines_in:
                if '**' in line:
                    continue
                if '*System' in line:
                    continue
                if '*Node' in line:
                    f.write('*Node, nset=Nall\n')
                    continue
                if '*Element' in line:
                    f.write('*User Element,Nodes=8,Type=U3,Iproperties=1,Properties=5,Coordinates=3,Unsymm\n')
                    f.write('1,2,3\n')
                    f.write('*Element, type=U3\n')
                    element_bool = True
                    continue
                if element_bool and '*' in line:
                    element_bool = False
                    f.write('*Element, type=C3D8, elset=ElDummy\n')
                    for el_line in element_lines:
                        words = el_line.split(',')
                        new_line = str(int(words[0]) + 10000) + ',' + \
                                   words[1] + ',' + words[2] + ',' + words[3] + ',' + words[4] + ',' + \
                                   words[5] + ',' + words[6] + ',' + words[7] + ',' + words[8]
                        f.write(new_line)
                if element_bool:
                    element_lines.append(line)

                f.write(line)

            #
            # now assign the material properties
            #
            for m in range(self.nx):
                for n in range(self.nz):
                    direction = self.locations[m][n]

                    if direction == 0:
                        Mx = 0.0
                        My = 0.0
                        Mz = 0.0
                    elif direction == 1:
                        Mx = -100.0e3
                        My = 0.0#100.0e3
                        Mz = 0.0
                    elif direction == 2:
                        Mx = 100.0e3
                        My = 0.0
                        Mz = 0.0
                    elif direction == 3:
                        Mx = 0.0
                        My = -100.0e3
                        Mz = 0.0
                    elif direction == 4:
                        Mx = -100.0e3
                        My = 0.0
                        Mz = 0.0
                    else:
                        print('Big Problem')

                    set_name = 'Structure_ElSet-' + str(m) + '-' + str(n)
                    f.write('*uel property, elset=' + set_name + '\n')
                    f.write(str(Gshear) + ',' + str(Kbulk) + ',' + str(Mx) + ',' + str(My) + ',' +
                            str(Mz) + ',' + str(8) + '\n')

            f.write('*Solid section, elset=ElDummy, material=Material-2\n')
            f.write('*Material, name=Material-1\n')
            f.write('*Elastic\n')
            f.write(str(1.e-20) + '\n')
            f.write('*Conductivity\n')
            f.write(str(1.0) + '\n')
            f.write('*Density\n')
            f.write(str(1.0) + '\n')
            f.write('*Material, name=Material-2\n')
            f.write('*Elastic\n')
            f.write(str(1.e-20) + '\n')
            f.write('*Density\n')
            f.write(str(2500.0) + '\n')
            f.write('*Amplitude, name=magFieldMagAmp\n')
            f.write('0.0,0.0,1.,0.05\n')
            f.write('*Step, name=deform, nlgeom=yes, inc=10000\n')
            f.write('*Static\n')
            f.write('0.2,1.,1.e-6,1.\n')
            f.write('*Controls, Parameters=Time Incrementation\n')
            f.write(',,,,,,,10,,,,,,\n')
            f.write('*Dload\n')
            f.write(', GRAV, 10., 0., -1., 0.\n')
            f.write('*Field, variable=1\n')
            f.write('Structure_Nall,0.0\n')
            f.write('*Field, variable=2\n')
            f.write('Structure_Nall,1.0\n')
            f.write('*Field, variable=3\n')
            f.write('Structure_Nall,0.0\n')
            f.write('*Field, variable=4, amplitude=magFieldMagAmp\n')
            f.write('Structure_Nall,1.0\n')
            f.write('*Output, field\n')
            f.write('*node output, nset=Nall\n')
            f.write('u\n')
            f.write('*element output, elset=ElDummy\n')
            f.write('le,ne\n')
            f.write('*End Step\n')

    def create_job(self):
        mdb.JobFromInputFile(name=self.job_name,
                             inputFileName=self.job_name + '.inp',
                             userSubroutine='uMag.for')
        mdb.jobs[self.job_name].submit()
        mdb.jobs[self.job_name].waitForCompletion()

        if mdb.jobs[self.job_name].status == ABORTED or mdb.jobs[self.job_name] == TERMINATED:
            return True
        else:
            return False

    def post_process(self, aborted):

        try:
            odb = session.openOdb(name=self.job_name + '.odb')
        except RunTimeError:
            pass

        output_file = 'output-' + self.job_name + '.txt'

        with open(output_file, 'w+') as f:
            f.write('*Initial Position\n')
            if aborted:
                f.write('*Simulation failed\n')
            if len(odb.steps['deform'].frames) == 0:
                f.write('*Simulation failed\n')

            assembly = odb.rootAssembly

            inst = assembly.instances['PART-1-1']

            nodes = inst.nodeSets['STRUCTURE_TOP-FACE']

            frame = odb.steps['deform'].frames[-1]

            u = frame.fieldOutputs['U'].getSubset(region=nodes)

            for node in nodes.nodes:
                f.write(str(node.label) + ',' + str(node.coordinates[0]) + ',' +
                        str(node.coordinates[1]) + ',' + str(node.coordinates[2]) + '\n')

            f.write('*Displacements\n')

            for v in u.values:
                f.write(str(v.nodeLabel) + ',' + str(v.data[0]) + ',' + str(v.data[1]) +
                        ',' + str(v.data[2]) + '\n')

        odb.close


def main(gene_file, job_name, length, height, depth):
    model1 = HardMagneticSoftMaterial(gene_file, job_name, length, height, depth)
    model1.generate_geometry()
    model1.partition_geometry()
    model1.create_assembly()
    model1.mesh_structure()
    model1.create_material_section()
    model1.assign_field_values()
    model1.assign_boundary_conditions()
    model1.write_input_file()
    model1.modify_input_file()
    aborted = model1.create_job()
    model1.post_process(aborted)


if __name__ == '__main__':

    args = sys.argv

    my_args = args[8:]

    gene_file_in = my_args[0]
    job_name_in = my_args[1]

    length_in = 0.025
    height_in = 0.00025
    depth_in = 0.01

    main(gene_file_in, job_name_in, length_in, height_in, depth_in)
