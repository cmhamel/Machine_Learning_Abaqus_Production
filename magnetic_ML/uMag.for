************************************************************************
!
!
! Solution variables (or nodal variables) are the displacements, with the
!  referential magnetic field applied using a predefined field.  DOF's 1,
!  2, and 3 are the displacements.
!
! Material behavior is Neo-Hookean rubber elasticity.
!
! This subroutine is for a three-dimensional 8 node isoparametric
!  element as shown below with 1pt (reduced) or 8pt (full) Gauss
!  integration.
!
! In order to avoid locking for the fully-integrated element, we
!  use the F-bar method of de Souza Neto (1996).
!
!  Mechanical, traction- and pressure-type boundary conditions 
!   may be applied to the dummy mesh using the Abaqus built-in 
!   commands *Dload or *Dsload.
!
!
!  8-node     8-----------7
!  brick     /|          /|       zeta
!           / |         / |       
!          5-----------6  |       |     eta
!          |  |        |  |       |   /
!          |  |        |  |       |  /
!          |  4--------|--3       | /
!          | /         | /        |/
!          |/          |/         O--------- xi
!          1-----------2        origin at cube center
!
!     Face numbering follows:
!       Face 1 = nodes 1,2,3,4
!       Face 2 = nodes 5,8,7,6
!       Face 3 = nodes 1,5,6,2
!       Face 4 = nodes 2,6,7,3
!       Face 5 = nodes 3,7,8,4
!       Face 6 = nodes 4,8,5,1
!
! Ruike Zhao 2017
***********************************************************************
!
! User element statement in the input file:
!  *User Element,Nodes=8,Type=U3,Iproperties=<nJProps>,Properties=<nProps>,Coordinates=3,Unsymm
!  1,2,3
!
!     Material Properties Vector
!     --------------------------------------------------------------
!     Gshear   = props(1) ! Shear modulus
!     Kbulk    = props(2) ! Bulk modulus
!     refMagMoment(1,1) = props(3) !
!     refMagMoment(2,1) = props(4) ! reference magnetic moment vector
!     refMagMoment(3,1) = props(5) !
!
***********************************************************************
!
      SUBROUTINE UEL(RHS,AMATRX,SVARS,ENERGY,NDOFEL,NRHS,NSVARS,
     +     PROPS,NPROPS,coords,MCRD,NNODE,Uall,DUall,Vel,Accn,JTYPE,
     +     TIME,DTIME,KSTEP,KINC,JELEM,PARAMS,NDLOAD,JDLTYP,ADLMAG,
     +     PREDEF,NPREDF,LFLAGS,MLVARX,DDLMAG,MDLOAD,PNEWDT,JPROPS,
     +     NJPROP,PERIOD)
!
      IMPLICIT NONE
!
!*     VARIABLES DEFINED IN UEL, PASSED BACK TO ABAQUS
!
      REAL(8) :: RHS,AMATRX,SVARS,ENERGY
!
!*     VARIABLES PASSED INTO UEL 
!
      REAL(8) :: PROPS,coords,Uall,DUall,Vel,Accn,TIME,
     1  DTIME,PARAMS,ADLMAG,PREDEF,DDLMAG,PNEWDT,PERIOD
      INTEGER :: NDOFEL,NRHS,NSVARS,NPROPS,MCRD,NNODE,JTYPE,KSTEP,KINC,
     1  JELEM,NDLOAD,JDLTYP,NPREDF,LFLAGS,MLVARX,MDLOAD,JPROPS,NJPROP
!
      DIMENSION RHS(MLVARX,*),AMATRX(NDOFEL,NDOFEL),PROPS(*),
     1 SVARS(*),ENERGY(8),coords(MCRD,NNODE),Uall(NDOFEL),
     2 DUall(MLVARX,*),Vel(NDOFEL),Accn(NDOFEL),TIME(2),PARAMS(*),
     3 JDLTYP(MDLOAD,*),ADLMAG(MDLOAD,*),DDLMAG(MDLOAD,*),
     4 PREDEF(2,NPREDF,NNODE),LFLAGS(*),JPROPS(*)
!
!
      ! Internally defined variables
      !
      integer nDim,nInt,nIntS


      ! Obtain the number of integration points
      !
      nInt = jprops(1)
!
!     Call user element
      if(jtype.eq.3) then
         !
         ! This is a 3D analysis
         !
         nDim = 3
         call U3D8(RHS,AMATRX,SVARS,ENERGY,NDOFEL,NRHS,NSVARS,
     +        PROPS,NPROPS,coords,MCRD,NNODE,Uall,DUall,Vel,Accn,JTYPE,
     +        TIME,DTIME,KSTEP,KINC,JELEM,PARAMS,NDLOAD,JDLTYP,ADLMAG,
     +        PREDEF,NPREDF,LFLAGS,MLVARX,DDLMAG,MDLOAD,PNEWDT,JPROPS,
     +        NJPROP,PERIOD,
     +        nDim,nInt,nIntS)
!       
      else
         write(*,*) 'Only element U3 allowed, currently jtype=',jtype
         call xit
!
      endif
      !
      ! Done with this element, RHS and AMATRX already returned
      !  as output from the specific element routine called
      !
      return
      end subroutine uel

************************************************************************
************************************************************************

      subroutine U3D8(RHS,AMATRX,SVARS,ENERGY,NDOFEL,NRHS,NSVARS,
     +     PROPS,NPROPS,coords,MCRD,NNODE,Uall,DUall,Vel,Accn,JTYPE,
     +     TIME,DTIME,KSTEP,KINC,JELEM,PARAMS,NDLOAD,JDLTYP,ADLMAG,
     +     PREDEF,NPREDF,LFLAGS,MLVARX,DDLMAG,MDLOAD,PNEWDT,JPROPS,
     +     NJPROP,PERIOD,
     +     nDim,nInt,nIntS)
!
      IMPLICIT NONE
!
!     VARIABLES DEFINED IN UEL, PASSED BACK TO ABAQUS
!
      REAL(8) :: RHS,AMATRX,SVARS,ENERGY
!
!     VARIABLES PASSED INTO UEL 
!
      REAL(8) :: PROPS,coords,Uall,DUall,Vel,Accn,TIME,
     1  DTIME,PARAMS,ADLMAG,PREDEF,DDLMAG,PNEWDT,PERIOD
      INTEGER :: NDOFEL,NRHS,NSVARS,NPROPS,MCRD,NNODE,JTYPE,KSTEP,KINC,
     1  JELEM,NDLOAD,JDLTYP,NPREDF,LFLAGS,MLVARX,MDLOAD,JPROPS,NJPROP
!
      DIMENSION RHS(MLVARX,*),AMATRX(NDOFEL,NDOFEL),PROPS(*),
     1 SVARS(*),ENERGY(8),coords(MCRD,NNODE),Uall(NDOFEL),
     2 DUall(MLVARX,*),Vel(NDOFEL),Accn(NDOFEL),TIME(2),PARAMS(*),
     3 JDLTYP(MDLOAD,*),ADLMAG(MDLOAD,*),DDLMAG(MDLOAD,*),
     4 PREDEF(2,NPREDF,NNODE),LFLAGS(*),JPROPS(*)
!
!     Internally defined variables
      real*8 u(nNode,3),du(nNode,ndofel),phi(nNode),coordsC(mcrd,nNode)
      real*8 uOld(nNode,ndofel),u_t(nNode,ndofel)
      real*8 dPhi(nNode)

      integer i,j,k,l,m,n,nIntPt,nDim,intpt,a1,b1,a11,b11,face
      integer nInt,ii,a12,b12,pe,nSdv,stat,q,nIntV,nIntPtV,p,nIntS
      integer nlSdv,kk,IntS,faceFlag

      real*8 Iden(3,3),Le,Ru(3*nNode,1),tmp
      real*8 Kuu(3*nNode,3*nNode),sh0(nNode)
      real*8 dshxi(nNode,3),dsh0(nNode,3),dshC0(nNode,3),detMapJ0C
      real*8 Fc_tau(3,3),Fc_t(3,3),detFc_tau,detFc_t,w(nInt),D_tau(3,1)
      real*8 sh(nNode),detMapJ,dsh(nNode,3),detMapJC,Limit,umeror
      real*8 dshC(nNode,3),F_tau(3,3)
      real*8 F_t(3,3),detF_tau,xi(nInt,3),detF,T_tau(3,3),bodyForce(3)
      real*8 SpUUMod(3,3,3,3),bodyCharge,Gmat(9,3*nNode),Amat(9,9)
      real*8 BodyForceRes(3*nNode,1),flux
      real*8 G0mat(9,3*nNode),Qmat(9,9),dA,Nvec(1,nNode),AmatPhiU(3,9)
      real*8 xLocal(nIntS),yLocal(nIntS),zLocal(nIntS),wS(nIntS),detF_t
      real*8 Kuphi(3*nNode,nNode),Kphiu(nNode,3*nNode),AmatUPhi(9,3)
      real*8 SpPhiUMod(3,3,3),SpPhiPhiMod(3,3),SpUPhiMod(3,3,3),detMapJ0
      real*8 ER(3,1),magFieldVec(3,1),magFieldMag,Svec(9,1)

      real*8 zero,one,two,half,Pi,three,third
      parameter(zero=0.d0,one=1.d0,two=2.d0,half=0.5d0,Pi=3.141592653d0,
     +     three=3.d0,third=1.d0/3.d0)

      ! Identity tensor
      !
      call onem(Iden)


      ! Initialize the residual and tangent matrices to zero.
      !
      Ru  = zero
      Kuu = zero
      Energy = zero

      !coords = 0.001 * coords
      ! Obtain nodal displacements and magnetic field information
      !
      k = 0
      do i=1,nNode
         do j=1,nDim
            k = k + 1
            u(i,j) = Uall(k)
            du(i,j) = DUall(k,1)
            uOld(i,j) = u(i,j) - du(i,j)
         enddo
      enddo

      ! Obtain the magnetic field vector using a predefined field
      !  that is implemented through the input file.  Specifically,
      !  "*Field, variable=1,2,3,4" provides the magnetic field
      !  vector components x, y, z, and amplitude.  Since it is
      !  uniform, we only take the value at the first node.
      !
      magFieldVec(1,1) = predef(1,2,1)
      magFieldVec(2,1) = predef(1,3,1)
      magFieldVec(3,1) = predef(1,4,1)
      magFieldMag = predef(1,5,1)
      magFieldVec = magFieldMag*magFieldVec


      ! Obtain current nodal coordinates
      !
      do i=1,nNode
         do j=1,nDim
            coordsC(j,i) = coords(j,i) + u(i,j)
         enddo
      enddo
      !
      ! Impose any time-stepping changes on the increments of
      !  displacement if you want (based on element diagonal)
      !
      Le = dsqrt(((coordsC(1,1)-coordsC(1,7))**two) + 
     +     ((coordsC(2,1)-coordsC(2,7))**two) +
     +     ((coordsC(3,1)-coordsC(3,7))**two))
      !
      do i=1,nNode
         do j=1,nDim
            if(dabs(du(i,j)).gt.10.d0*Le) then
               pnewdt = 0.5
               return
            endif
         enddo
      enddo

      !----------------------------------------------------------------
      ! 
      ! Take this opportunity to perform calculations at the element
      !  centroid.  Here, check for hourglass stabilization and get
      !  the deformation gradient for use in the `F-bar' method.
      !
      ! Reference for the F-bar method:
      !  de Souza Neto, E.A., Peric, D., Dutko, M., Owen, D.R.J., 1996.
      !  Design of simple low order finite elements for large strain
      !  analysis of nearly incompressible solids. International Journal
      !  of Solids and Structures, 33, 3277-3296.
      !
      !
      ! Obtain shape functions and their local gradients at the element
      !  centroid, that means xi=eta=zeta=0.0, and nIntPt=1
      !
      if(nNode.eq.8) then
         call calcShape3DLinear(1,zero,1,sh0,dshxi)
      else
         write(*,*) 'Incorrect number of nodes: nNode.ne.8'
         call xit
      endif


      ! Map shape functions from local to global reference coordinate system
      !
      call mapShape3D(nNode,dshxi,coords,dsh0,detMapJ0,stat)
      if(stat.eq.0) then
         pnewdt = 0.5
         return
      endif



      ! Map shape functions from local to global current coordinate system
      !
      call mapShape3D(nNode,dshxi,coordsC,dshC0,detMapJ0C,stat)
      if(stat.eq.0) then
         pnewdt = 0.5
         return
      endif


      ! Calculate the deformation gradient at the element centroid
      !  at the the beginning and end of the increment for use in 
      !  the `F-bar' method
      !
      Fc_tau = Iden
      Fc_t = Iden
      do i=1,nDim
         do j=1,nDim
            do k=1,nNode
               Fc_tau(i,j) = Fc_tau(i,j) + dsh0(k,j)*u(k,i)
               Fc_t(i,j) = Fc_t(i,j) + dsh0(k,j)*uOld(k,i)
            enddo
         enddo
      enddo
      call mdet(Fc_tau,detFc_tau)
      call mdet(Fc_t,detFc_t)
      !
      ! With the deformation gradient known at the element centroid
      !  we are now able to implement the `F-bar' method later
      !
      !----------------------------------------------------------------




      !----------------------------------------------------------------
      ! Begin the loop over integration points
      !
      ! Obtain integration point local coordinates and weights
      !
      if(nInt.eq.1) then
         call xint3D1pt(xi,w,nIntPt) ! 1-pt integration, nInt=1
      elseif(nInt.eq.8) then
         call xint3D8pt(xi,w,nIntPt) ! 8-pt integration, nInt=8
      else
         write(*,*) 'Invalid number of int points, nInt=',nInt
         call xit
      endif


      ! Loop over integration points
      !
      do intpt=1,nIntPt

         ! Obtain shape functions and their local gradients
         !
         if(nNode.eq.8) then
            call calcShape3DLinear(nIntPt,xi,intpt,sh,dshxi)
         else
            write(*,*) 'Incorrect number of nodes: nNode.ne.8'
            call xit
         endif


         ! Map shape functions from local to global reference coordinate system
         !
         call mapShape3D(nNode,dshxi,coords,dsh,detMapJ,stat)
         if(stat.eq.0) then
            pnewdt = 0.5
            return
         endif


         ! Map shape functions from local to global current coordinate system
         !
         call mapShape3D(nNode,dshxi,coordsC,dshC,detMapJC,stat)
         if(stat.eq.0) then
            pnewdt = 0.5
            return
         endif

         ! Obtain, and modify the deformation gradient at this integration
         !  point.  Modify the deformation gradient for use in the `F-bar'
         !  method if needed.
         !
         F_tau = Iden
         F_t = Iden
         do i=1,nDim
            do j=1,nDim
               do k=1,nNode
                  F_tau(i,j) = F_tau(i,j) + dsh(k,j)*u(k,i)
                  F_t(i,j) = F_t(i,j) + dsh(k,j)*uOld(k,i)
               enddo
            enddo
         enddo
         !
         ! Modify the deformation gradient for the `F-bar' method
         !  only when using the 8 node fully integrated linear
         !  element, do not use the `F-bar' method for any other element
         !
         if((nNode.eq.8).and.(nInt.eq.8)) then
            call mdet(F_tau,detF_tau)
            call mdet(F_t,detF_t)
            F_tau = ((detFc_tau/detF_tau)**third)*F_tau
            F_t = ((detFc_tau/detF_tau)**third)*F_t
         endif
         call mdet(F_tau,detF)



         ! Perform the time integration at this integ. point to compute
         !  all the specific forms and parameters needed for the solution
         !
         call integ(props,nprops,dtime,
     +        F_tau,magFieldVec,
     +        T_tau,
     +        SpUUMod,stat)
         !
         if(stat.eq.0) then
            pnewdt = 0.5
            return
         endif



         ! Compute the matrices that contain the shape functions
         ! and derivatives of the shape functions
         !
         do kk=1,nNode
            Nvec(1,kk) = sh(kk)
         enddo
         !
         !
         Gmat = zero
         do kk=1,nNode
            Gmat(1,1+nDim*(kk-1)) = dshC(kk,1)
            Gmat(2,2+nDim*(kk-1)) = dshC(kk,1)
            Gmat(3,3+nDim*(kk-1)) = dshC(kk,1)
            Gmat(4,1+nDim*(kk-1)) = dshC(kk,2)
            Gmat(5,2+nDim*(kk-1)) = dshC(kk,2)
            Gmat(6,3+nDim*(kk-1)) = dshC(kk,2)
            Gmat(7,1+nDim*(kk-1)) = dshC(kk,3)
            Gmat(8,2+nDim*(kk-1)) = dshC(kk,3)
            Gmat(9,3+nDim*(kk-1)) = dshC(kk,3)
         enddo
         !
         G0mat = zero
         do kk=1,nNode
            G0mat(1,1+nDim*(kk-1)) = dshC0(kk,1)
            G0mat(2,2+nDim*(kk-1)) = dshC0(kk,1)
            G0mat(3,3+nDim*(kk-1)) = dshC0(kk,1)
            G0mat(4,1+nDim*(kk-1)) = dshC0(kk,2)
            G0mat(5,2+nDim*(kk-1)) = dshC0(kk,2)
            G0mat(6,3+nDim*(kk-1)) = dshC0(kk,2)
            G0mat(7,1+nDim*(kk-1)) = dshC0(kk,3)
            G0mat(8,2+nDim*(kk-1)) = dshC0(kk,3)
            G0mat(9,3+nDim*(kk-1)) = dshC0(kk,3)
         enddo




         ! Compute/update the displacement residual vector
         !
         Svec(1,1) = T_tau(1,1)
         Svec(2,1) = T_tau(2,1)
         Svec(3,1) = T_tau(3,1)
         Svec(4,1) = T_tau(1,2)
         Svec(5,1) = T_tau(2,2)
         Svec(6,1) = T_tau(3,2)
         Svec(7,1) = T_tau(1,3)
         Svec(8,1) = T_tau(2,3)
         Svec(9,1) = T_tau(3,3)
         !
         bodyForce = zero ! the body force vector may be specified here
         !
         BodyForceRes = zero
         do kk=1,nNode
            BodyForceRes(1+nDim*(kk-1),1) = sh(kk)*bodyForce(1)
            BodyForceRes(2+nDim*(kk-1),1) = sh(kk)*bodyForce(2)
            BodyForceRes(3+nDim*(kk-1),1) = sh(kk)*bodyForce(3)
         enddo
         !
         Ru = Ru + detmapJC*w(intpt)*
     +        (
     +        -matmul(transpose(Gmat),Svec)
     +        + BodyForceRes
     +        )



         ! Compute/update the displacement tangent matrix
         !
         Amat = zero
         Amat(1,1) = SpUUMod(1,1,1,1)
         Amat(1,2) = SpUUMod(1,1,2,1)
         Amat(1,3) = SpUUMod(1,1,3,1)
         Amat(1,4) = SpUUMod(1,1,1,2)
         Amat(1,5) = SpUUMod(1,1,2,2)
         Amat(1,6) = SpUUMod(1,1,3,2)
         Amat(1,7) = SpUUMod(1,1,1,3)
         Amat(1,8) = SpUUMod(1,1,2,3)
         Amat(1,9) = SpUUMod(1,1,3,3)
         Amat(2,1) = SpUUMod(2,1,1,1)
         Amat(2,2) = SpUUMod(2,1,2,1)
         Amat(2,3) = SpUUMod(2,1,3,1)
         Amat(2,4) = SpUUMod(2,1,1,2)
         Amat(2,5) = SpUUMod(2,1,2,2)
         Amat(2,6) = SpUUMod(2,1,3,2)
         Amat(2,7) = SpUUMod(2,1,1,3)
         Amat(2,8) = SpUUMod(2,1,2,3)
         Amat(2,9) = SpUUMod(2,1,3,3)
         Amat(3,1) = SpUUMod(3,1,1,1)
         Amat(3,2) = SpUUMod(3,1,2,1)
         Amat(3,3) = SpUUMod(3,1,3,1)
         Amat(3,4) = SpUUMod(3,1,1,2)
         Amat(3,5) = SpUUMod(3,1,2,2)
         Amat(3,6) = SpUUMod(3,1,3,2)
         Amat(3,7) = SpUUMod(3,1,1,3)
         Amat(3,8) = SpUUMod(3,1,2,3)
         Amat(3,9) = SpUUMod(3,1,3,3)
         Amat(4,1) = SpUUMod(1,2,1,1)
         Amat(4,2) = SpUUMod(1,2,2,1)
         Amat(4,3) = SpUUMod(1,2,3,1)
         Amat(4,4) = SpUUMod(1,2,1,2)
         Amat(4,5) = SpUUMod(1,2,2,2)
         Amat(4,6) = SpUUMod(1,2,3,2)
         Amat(4,7) = SpUUMod(1,2,1,3)
         Amat(4,8) = SpUUMod(1,2,2,3)
         Amat(4,9) = SpUUMod(1,2,3,3)
         Amat(5,1) = SpUUMod(2,2,1,1)
         Amat(5,2) = SpUUMod(2,2,2,1)
         Amat(5,3) = SpUUMod(2,2,3,1)
         Amat(5,4) = SpUUMod(2,2,1,2)
         Amat(5,5) = SpUUMod(2,2,2,2)
         Amat(5,6) = SpUUMod(2,2,3,2)
         Amat(5,7) = SpUUMod(2,2,1,3)
         Amat(5,8) = SpUUMod(2,2,2,3)
         Amat(5,9) = SpUUMod(2,2,3,3)
         Amat(6,1) = SpUUMod(3,2,1,1)
         Amat(6,2) = SpUUMod(3,2,2,1)
         Amat(6,3) = SpUUMod(3,2,3,1)
         Amat(6,4) = SpUUMod(3,2,1,2)
         Amat(6,5) = SpUUMod(3,2,2,2)
         Amat(6,6) = SpUUMod(3,2,3,2)
         Amat(6,7) = SpUUMod(3,2,1,3)
         Amat(6,8) = SpUUMod(3,2,2,3)
         Amat(6,9) = SpUUMod(3,2,3,3)
         Amat(7,1) = SpUUMod(1,3,1,1)
         Amat(7,2) = SpUUMod(1,3,2,1)
         Amat(7,3) = SpUUMod(1,3,3,1)
         Amat(7,4) = SpUUMod(1,3,1,2)
         Amat(7,5) = SpUUMod(1,3,2,2)
         Amat(7,6) = SpUUMod(1,3,3,2)
         Amat(7,7) = SpUUMod(1,3,1,3)
         Amat(7,8) = SpUUMod(1,3,2,3)
         Amat(7,9) = SpUUMod(1,3,3,3)
         Amat(8,1) = SpUUMod(2,3,1,1)
         Amat(8,2) = SpUUMod(2,3,2,1)
         Amat(8,3) = SpUUMod(2,3,3,1)
         Amat(8,4) = SpUUMod(2,3,1,2)
         Amat(8,5) = SpUUMod(2,3,2,2)
         Amat(8,6) = SpUUMod(2,3,3,2)
         Amat(8,7) = SpUUMod(2,3,1,3)
         Amat(8,8) = SpUUMod(2,3,2,3)
         Amat(8,9) = SpUUMod(2,3,3,3)
         Amat(9,1) = SpUUMod(3,3,1,1)
         Amat(9,2) = SpUUMod(3,3,2,1)
         Amat(9,3) = SpUUMod(3,3,3,1)
         Amat(9,4) = SpUUMod(3,3,1,2)
         Amat(9,5) = SpUUMod(3,3,2,2)
         Amat(9,6) = SpUUMod(3,3,3,2)
         Amat(9,7) = SpUUMod(3,3,1,3)
         Amat(9,8) = SpUUMod(3,3,2,3)
         Amat(9,9) = SpUUMod(3,3,3,3)


         Qmat = zero
         Qmat(1,1) = third*(Amat(1,1)+Amat(1,5)+Amat(1,9)) 
     +        - (two/three)*T_tau(1,1)
         Qmat(2,1) = third*(Amat(2,1)+Amat(2,5)+Amat(2,9))
     +        - (two/three)*T_tau(2,1)
         Qmat(3,1) = third*(Amat(3,1)+Amat(3,5)+Amat(3,9))
     +        - (two/three)*T_tau(3,1)
         Qmat(4,1) = third*(Amat(4,1)+Amat(4,5)+Amat(4,9))
     +        - (two/three)*T_tau(1,2)
         Qmat(5,1) = third*(Amat(5,1)+Amat(5,5)+Amat(5,9))
     +        - (two/three)*T_tau(2,2)
         Qmat(6,1) = third*(Amat(6,1)+Amat(6,5)+Amat(6,9))
     +        - (two/three)*T_tau(3,2)
         Qmat(7,1) = third*(Amat(7,1)+Amat(7,5)+Amat(7,9))
     +        - (two/three)*T_tau(1,3)
         Qmat(8,1) = third*(Amat(8,1)+Amat(8,5)+Amat(8,9))
     +        - (two/three)*T_tau(2,3)
         Qmat(9,1) = third*(Amat(9,1)+Amat(9,5)+Amat(9,9))
     +        - (two/three)*T_tau(3,3)
         Qmat(1,5) = Qmat(1,1)
         Qmat(2,5) = Qmat(2,1)
         Qmat(3,5) = Qmat(3,1)
         Qmat(4,5) = Qmat(4,1)
         Qmat(5,5) = Qmat(5,1)
         Qmat(6,5) = Qmat(6,1)
         Qmat(7,5) = Qmat(7,1)
         Qmat(8,5) = Qmat(8,1)
         Qmat(9,5) = Qmat(9,1)
         Qmat(1,9) = Qmat(1,1)
         Qmat(2,9) = Qmat(2,1)
         Qmat(3,9) = Qmat(3,1)
         Qmat(4,9) = Qmat(4,1)
         Qmat(5,9) = Qmat(5,1)
         Qmat(6,9) = Qmat(6,1)
         Qmat(7,9) = Qmat(7,1)
         Qmat(8,9) = Qmat(8,1)
         Qmat(9,9) = Qmat(9,1)
         

         if((nNode.eq.8).and.(nInt.eq.8)) then
            !
            ! This is the tangent using the F-bar method with the
            !  8 node fully integrated linear element
            !
            Kuu = Kuu + detMapJC*w(intpt)*
     +           (
     +           matmul(matmul(transpose(Gmat),Amat),Gmat)
     +           + matmul(transpose(Gmat),matmul(Qmat,(G0mat-Gmat)))
     +           )
         else
            !
            ! This is the tangent NOT using the F-bar method with all
            !  other elements
            !
            Kuu = Kuu + detMapJC*w(intpt)*
     +           (
     +           matmul(matmul(transpose(Gmat),Amat),Gmat)
     +           )
         endif


      enddo
      !
      ! End the loop over integration points
      !----------------------------------------------------------------

      ! Return the RHS vector and the Stiffness matrix.
      !
      rhs(1:24,1) = Ru(:,1)
      amatrx = Kuu
      !
      ! End return of RHS and AMATRX
      !----------------------------------------------------------------

      return
      end subroutine U3D8

************************************************************************

      subroutine integ(props,nprops,dtime,
     +        F_tau,magFieldVec,
     +        T_tau,
     +        SpUUMod,stat)


      implicit none

      integer i,j,k,l,m,n,stat,nprops

      real*8 Iden(3,3),F_tau(3,3),FT(3,3),FinvT(3,3),C_tau(3,3),detC
      real*8 Cinv(3,3),trC,I1bar,detF,Finv(3,3),magFieldVec(3,1)
      real*8 TR_tau(3,3),T_tau(3,3),dTRdF(3,3,3,3),Gshear,Kbulk,G0,trAC
      real*8 SpUUmod(3,3,3,3),lamL,aux
      real*8 props(nprops),dtime,AC(3,3),lamBar
      real*8 Fbar(3,3),AFt(3,3),trFAFt,FAFt(3,3)
      real*8 FA(3,3),dGdF(3,3),refMagMoment(3,1),TR_eq(3,3),TR_mag(3,3)
      real*8 MB(3,3),MBFinvT(3,3),FMBFinvT(3,3)


      real*8 zero,one,two,three,third,half,nine
      parameter(zero=0.d0,one=1.d0,two=2.d0,three=3.d0,third=1.d0/3.d0,
     +     half=1.d0/2.d0,nine=9.d0)


      ! Identity matrix
      !
      call onem(Iden)

      ! Obtain material parameters
      !
      Gshear = props(1) !* 1e-6
      Kbulk  = props(2) !* 1e-6
      refMagMoment(1,1) = props(3)
      refMagMoment(2,1) = props(4)
      refMagMoment(3,1) = props(5)

      ! Compute the determinant, the inverse, the transpose, 
      !  and the inverse transpose of the deformation gradient
      !
      call matInv3D(F_tau,Finv,detF,stat)
      FT = transpose(F_tau)
      FinvT = transpose(Finv)
      Fbar = (detF**(-two/three))*F_tau


      ! Compute the right Cauchy-Green tensor, its inverse, 
      !  its determinant, and its trace
      !
      C_tau = matmul(transpose(F_tau),F_tau)
      call matInv3D(C_tau,Cinv,detC,stat)
      trC = C_tau(1,1) + C_tau(2,2) + C_tau(3,3)
 
      ! Compute the equilibrium 1st Piola stress, including the 
      !  magnetic contribution
      !
      TR_tau = (detF**(-two/three))*Gshear*(F_tau - third*trC*FinvT)
     +     + Kbulk*detF*(detF - one)*FinvT
     +     - matmul(magFieldVec,transpose(refMagMoment))

      ! Compute the total Cauchy stress
      !
      T_tau = (one/detF)*matmul(TR_tau,transpose(F_tau))

      ! Calculate the material tangent modulus
      !
      dTRdF = zero
      do i=1,3
         do j=1,3
            do k=1,3
               do l=1,3
                  dTRdF(i,j,k,l) = dTRdF(i,j,k,l)
     +                 + (detF**(-two/three))*Gshear*
     +                 (
     +                 (-two/three)*F_tau(i,j)*Finv(l,k)
     +                 + (two/9.d0)*trC*Finv(j,i)*Finv(l,k)
     +                 + Iden(i,k)*Iden(j,l)
     +                 + third*trC*Finv(l,i)*Finv(j,k)
     +                 - (two/three)*Finv(j,i)*F_tau(k,l)
     +                 )
     +                 + detF*Kbulk*
     +                 (
     +                 (detF-one)*Finv(j,i)*Finv(l,k)
     +                 + detF*Finv(j,i)*Finv(l,k)
     +                 - (detF-one)*Finv(l,i)*Finv(j,k)
     +                 )
               enddo
            enddo
         enddo
      enddo


      ! Calculate the material spatial tangent modulus
      !
      SpUUMod = zero
      do i=1,3
         do j=1,3
            do k=1,3
               do l=1,3
                  do m=1,3
                     do n=1,3
                        SpUUMod(i,j,k,l) = SpUUMod(i,j,k,l) +
     +                      (dTRdF(i,m,k,n)*F_tau(j,m)*F_tau(l,n))/detF
                     enddo
                  enddo
               enddo
            enddo
         enddo
      enddo


      return
      end subroutine integ

!****************************************************************************
!     Element subroutines
!****************************************************************************

      subroutine xint3D1pt(xi,w,nIntPt)
      
      ! This subroutine will get the integration point locations
      !  and corresponding gauss quadrature weights for 3D elements
      !  using a 2 gauss points for integration
      !
      !  xi(nIntPt,3): xi,eta,zeta coordinates for the integration pts
      !  w(nIntPt):    corresponding integration weights
      
      implicit none

      integer nIntPt,nDim

      real*8 xi(1,3),w(1)


      ! Init
      !
      w = 0.d0
      xi = 0.d0


      ! Number of Gauss points
      !
      nIntPt = 1


      ! Gauss weights
      !
      w(1) = 8.d0
      

      ! Gauss pt locations in master element
      !
      xi(1,1) = 0.d0
      xi(1,2) = 0.d0
      xi(1,3) = 0.d0

      return
      end subroutine xint3D1pt
     
************************************************************************

      subroutine xint3D8pt(xi,w,nIntPt)
      
      ! This subroutine will get the integration point locations
      !  and corresponding gauss quadrature weights for 3D elements
      !  using 8 gauss points for integration
      !
      !  xi(nIntPt,3): xi,eta,zeta coordinates for the integration pts
      !  w(nIntPt):    corresponding integration weights
      
      implicit none

      integer nIntPt,nDim

      real*8 xi(8,3),w(8)


      ! Init
      !
      w = 0.d0
      xi = 0.d0


      ! Number of Gauss points
      !
      nIntPt = 8


      ! Gauss weights
      !
      w(1) = 1.d0
      w(2) = 1.d0
      w(3) = 1.d0
      w(4) = 1.d0
      w(5) = 1.d0
      w(6) = 1.d0
      w(7) = 1.d0
      w(8) = 1.d0
      

      ! Gauss pt locations in master element
      !
      xi(1,1) = -dsqrt(1.d0/3.d0)
      xi(1,2) = -dsqrt(1.d0/3.d0)
      xi(1,3) = -dsqrt(1.d0/3.d0)
      xi(2,1) = dsqrt(1.d0/3.d0)
      xi(2,2) = -dsqrt(1.d0/3.d0)
      xi(2,3) = -dsqrt(1.d0/3.d0)
      xi(3,1) = -dsqrt(1.d0/3.d0)
      xi(3,2) = dsqrt(1.d0/3.d0)
      xi(3,3) = -dsqrt(1.d0/3.d0)
      xi(4,1) = dsqrt(1.d0/3.d0)
      xi(4,2) = dsqrt(1.d0/3.d0)
      xi(4,3) = -dsqrt(1.d0/3.d0)
      xi(5,1) = -dsqrt(1.d0/3.d0)
      xi(5,2) = -dsqrt(1.d0/3.d0)
      xi(5,3) = dsqrt(1.d0/3.d0)
      xi(6,1) = dsqrt(1.d0/3.d0)
      xi(6,2) = -dsqrt(1.d0/3.d0)
      xi(6,3) = dsqrt(1.d0/3.d0)
      xi(7,1) = -dsqrt(1.d0/3.d0)
      xi(7,2) = dsqrt(1.d0/3.d0)
      xi(7,3) = dsqrt(1.d0/3.d0)
      xi(8,1) = dsqrt(1.d0/3.d0)
      xi(8,2) = dsqrt(1.d0/3.d0)
      xi(8,3) = dsqrt(1.d0/3.d0)


      return
      end subroutine xint3D8pt

************************************************************************
************************************************************************

      subroutine calcShape3DLinear(nIntPt,xi_int,intpt,sh,dshxi)
      !
      !
      ! Calculate the shape functions and their derivatives at the
      ! given integration point in the master element
      !
      ! This subroutine uses a 8-node linear 3D element as shown
      !
      !      8-----------7
      !     /|          /|       zeta
      !    / |         / |       
      !   5-----------6  |       |     eta
      !   |  |        |  |       |   /
      !   |  |        |  |       |  /
      !   |  4--------|--3       | /
      !   | /         | /        |/
      !   |/          |/         O--------- xi
      !   1-----------2        origin at cube center
      !
      !
      ! sh(i) = shape function of node i at the intpt.
      ! dshxi(i,j) = derivative wrt j direction of shape fn of node i
      ! d2shxi(i,j,k) = derivatives wrt j and k of shape fn of node i

      implicit none

      integer intpt,nDim,nIntPt,i,j

      real*8 xi_int(nIntPt,3),sh(8),dshxi(8,3)
      real*8 d2shxi(8,3,3),xi,eta,zeta

      real*8 zero,one,two,half,fourth,eighth
      parameter(zero=0.d0,one=1.d0,two=2.d0,half=0.5d0,fourth=0.25d0,
     +     eighth=1.d0/8.d0)
      

      ! Location in the master element
      !
      xi = xi_int(intpt,1)
      eta = xi_int(intpt,2)
      zeta = xi_int(intpt,3)
      !
      ! The shape functions
      !
      sh(1) = eighth*(one - xi)*(one - eta)*(one - zeta)
      sh(2) = eighth*(one + xi)*(one - eta)*(one - zeta)
      sh(3) = eighth*(one + xi)*(one + eta)*(one - zeta)
      sh(4) = eighth*(one - xi)*(one + eta)*(one - zeta)
      sh(5) = eighth*(one - xi)*(one - eta)*(one + zeta)
      sh(6) = eighth*(one + xi)*(one - eta)*(one + zeta)
      sh(7) = eighth*(one + xi)*(one + eta)*(one + zeta)
      sh(8) = eighth*(one - xi)*(one + eta)*(one + zeta)
      !
      ! The first derivatives
      !
      dshxi(1,1) = -eighth*(one - eta)*(one - zeta)
      dshxi(1,2) = -eighth*(one - xi)*(one - zeta)
      dshxi(1,3) = -eighth*(one - xi)*(one - eta)
      dshxi(2,1) = eighth*(one - eta)*(one - zeta)
      dshxi(2,2) = -eighth*(one + xi)*(one - zeta)
      dshxi(2,3) = -eighth*(one + xi)*(one - eta)
      dshxi(3,1) = eighth*(one + eta)*(one - zeta)
      dshxi(3,2) = eighth*(one + xi)*(one - zeta)
      dshxi(3,3) = -eighth*(one + xi)*(one + eta)
      dshxi(4,1) = -eighth*(one + eta)*(one - zeta)
      dshxi(4,2) = eighth*(one - xi)*(one - zeta)
      dshxi(4,3) = -eighth*(one - xi)*(one + eta)
      dshxi(5,1) = -eighth*(one - eta)*(one + zeta)
      dshxi(5,2) = -eighth*(one - xi)*(one + zeta)
      dshxi(5,3) = eighth*(one - xi)*(one - eta)
      dshxi(6,1) = eighth*(one - eta)*(one + zeta)
      dshxi(6,2) = -eighth*(one + xi)*(one + zeta)
      dshxi(6,3) = eighth*(one + xi)*(one - eta)
      dshxi(7,1) = eighth*(one + eta)*(one + zeta)
      dshxi(7,2) = eighth*(one + xi)*(one + zeta)
      dshxi(7,3) = eighth*(one + xi)*(one + eta)
      dshxi(8,1) = -eighth*(one + eta)*(one + zeta)
      dshxi(8,2) = eighth*(one - xi)*(one + zeta)
      dshxi(8,3) = eighth*(one - xi)*(one + eta)
      !
      ! The second derivatives
      !
      d2shxi = zero
      d2shxi(1,1,2) = eighth*(one - zeta)
      d2shxi(1,2,1) = d2shxi(1,1,2)
      d2shxi(1,1,3) = eighth*(one - eta)
      d2shxi(1,3,1) = d2shxi(1,1,3)
      d2shxi(1,2,3) = eighth*(one - xi)
      d2shxi(1,3,2) = d2shxi(1,2,3)
      d2shxi(2,1,2) = -eighth*(one - zeta)
      d2shxi(2,2,1) = d2shxi(2,1,2)
      d2shxi(2,1,3) = -eighth*(one - eta)
      d2shxi(2,3,1) = d2shxi(2,1,3)
      d2shxi(2,2,3) = eighth*(one + xi)
      d2shxi(2,3,2) = d2shxi(2,2,3)
      d2shxi(3,1,2) = eighth*(one - zeta)
      d2shxi(3,2,1) = d2shxi(2,1,2)
      d2shxi(3,1,3) = -eighth*(one + eta)
      d2shxi(3,3,1) = d2shxi(2,1,3)
      d2shxi(3,2,3) = -eighth*(one + xi)
      d2shxi(3,3,2) = d2shxi(2,2,3)
      d2shxi(4,1,2) = -eighth*(one - zeta)
      d2shxi(4,2,1) = d2shxi(2,1,2)
      d2shxi(4,1,3) = eighth*(one + eta)
      d2shxi(4,3,1) = d2shxi(2,1,3)
      d2shxi(4,2,3) = -eighth*(one - xi)
      d2shxi(4,3,2) = d2shxi(2,2,3)
      d2shxi(5,1,2) = eighth*(one + zeta)
      d2shxi(5,2,1) = d2shxi(2,1,2)
      d2shxi(5,1,3) = -eighth*(one - eta)
      d2shxi(5,3,1) = d2shxi(2,1,3)
      d2shxi(5,2,3) = -eighth*(one - xi)
      d2shxi(5,3,2) = d2shxi(2,2,3)
      d2shxi(6,1,2) = eighth*(one + zeta)
      d2shxi(6,2,1) = d2shxi(2,1,2)
      d2shxi(6,1,3) = eighth*(one - eta)
      d2shxi(6,3,1) = d2shxi(2,1,3)
      d2shxi(6,2,3) = -eighth*(one + xi)
      d2shxi(6,3,2) = d2shxi(2,2,3)
      d2shxi(7,1,2) = eighth*(one + zeta)
      d2shxi(7,2,1) = d2shxi(2,1,2)
      d2shxi(7,1,3) = eighth*(one + eta)
      d2shxi(7,3,1) = d2shxi(2,1,3)
      d2shxi(7,2,3) = eighth*(one + xi)
      d2shxi(7,3,2) = d2shxi(2,2,3)
      d2shxi(8,1,2) = -eighth*(one + zeta)
      d2shxi(8,2,1) = d2shxi(2,1,2)
      d2shxi(8,1,3) = -eighth*(one + eta)
      d2shxi(8,3,1) = d2shxi(2,1,3)
      d2shxi(8,2,3) = eighth*(one - xi)
      d2shxi(8,3,2) = d2shxi(2,2,3)
      
      return
      end subroutine calcShape3DLinear

************************************************************************
************************************************************************

      subroutine mapShape3D(nNode,dshxi,coords,dsh,detMapJ,stat)
      !
      ! Map derivatives of shape fns from xi-eta-zeta domain
      !  to x-y-z domain.  This subroutine works for both 8-node
      !  linear and 20-node quadratic 3D elements.
      !
      implicit none

      integer i,j,k,nNode,ieror,stat

      real*8 dshxi(nNode,3),dsh(nNode,3),coords(3,nNode)
      real*8 mapJ(3,3),mapJ_inv(3,3),detmapJ

      real*8 zero,one,two,half,fourth,eighth
      parameter(zero=0.d0,one=1.d0,two=2.d0,half=0.5d0,fourth=0.25d0,
     +     eighth=1.d0/8.d0)
      

      ! Calculate the mapping Jacobian matrix:
      !
      mapJ = zero
      do i=1,3
        do j=1,3
          do k=1,nNode
              mapJ(i,j) = mapJ(i,j) + dshxi(k,i)*coords(j,k)
          end do
        end do
      end do


      ! Calculate the inverse and the determinant of Jacobian
      !
      call matInv3D(mapJ,mapJ_inv,detMapJ,stat)


      ! Calculate first derivatives wrt x, y, z
      !
      dsh = transpose(matmul(mapJ_inv,transpose(dshxi)))


      ! The second derivatives may be calculated.
      !

      return
      end subroutine mapShape3D

!****************************************************************************
!     Utility subroutines
!****************************************************************************

      subroutine matInv3D(A,A_inv,det_A,istat)
      !
      ! Returns A_inv, the inverse and det_A, the determinant
      ! Note that the det is of the original matrix, not the
      ! inverse
      !
      implicit none
      !
      integer istat
      !
      real*8 A(3,3),A_inv(3,3),det_A,det_A_inv


      istat = 1
      
      det_A = A(1,1)*(A(2,2)*A(3,3) - A(3,2)*A(2,3)) -
     +        A(2,1)*(A(1,2)*A(3,3) - A(3,2)*A(1,3)) +
     +        A(3,1)*(A(1,2)*A(2,3) - A(2,2)*A(1,3))
      
      if (det_A .le. 0.d0) then
        write(*,*) 'WARNING: subroutine matInv3D:'
        write(*,*) 'WARNING: det of mat=',det_A
        istat = 0
        return
      end if
          
      det_A_inv = 1.d0/det_A
        
      A_inv(1,1) = det_A_inv*(A(2,2)*A(3,3)-A(3,2)*A(2,3))
      A_inv(1,2) = det_A_inv*(A(3,2)*A(1,3)-A(1,2)*A(3,3))
      A_inv(1,3) = det_A_inv*(A(1,2)*A(2,3)-A(2,2)*A(1,3))
      A_inv(2,1) = det_A_inv*(A(3,1)*A(2,3)-A(2,1)*A(3,3))
      A_inv(2,2) = det_A_inv*(A(1,1)*A(3,3)-A(3,1)*A(1,3))
      A_inv(2,3) = det_A_inv*(A(2,1)*A(1,3)-A(1,1)*A(2,3))
      A_inv(3,1) = det_A_inv*(A(2,1)*A(3,2)-A(3,1)*A(2,2))
      A_inv(3,2) = det_A_inv*(A(3,1)*A(1,2)-A(1,1)*A(3,2))
      A_inv(3,3) = det_A_inv*(A(1,1)*A(2,2)-A(2,1)*A(1,2))
      

      return
      end subroutine matInv3D

!****************************************************************************

      subroutine mdet(A,det)
      !
      ! This subroutine calculates the determinant
      ! of a 3 by 3 matrix [A]
      !
      implicit none
      !
      real*8  A(3,3),det


      det = A(1,1)*A(2,2)*A(3,3) 
     +	  + A(1,2)*A(2,3)*A(3,1)
     +	  + A(1,3)*A(2,1)*A(3,2)
     +	  - A(3,1)*A(2,2)*A(1,3)
     +	  - A(3,2)*A(2,3)*A(1,1)
     +	  - A(3,3)*A(2,1)*A(1,2)


      return
      end subroutine mdet
	
!****************************************************************************

      subroutine onem(A)
      !
      ! This subroutine stores the identity matrix in the
      ! 3 by 3 matrix [A]
      !
      implicit none
      !
      integer i,j
      !
      real*8 A(3,3)


      do i=1,3
         do J=1,3
	    if (i .eq. j) then
              A(i,j) = 1.0
            else
              A(i,j) = 0.0
            end if
         end do
      end do


      return
      end subroutine onem
!
*****************************************************************************
