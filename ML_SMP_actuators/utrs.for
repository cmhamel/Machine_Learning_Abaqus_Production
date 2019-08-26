      subroutine utrs(shift,temp,dtemp,time,dtime,predef,dpred,
     +                statev,cmname,coords)

      include 'ABA_PARAM.INC'
      !
      character*80 cmname
      dimension shift(2),time(2),predef(1),dpred(1),statev(1)
      dimension coords(1)
      !
      ! my variables below
      !
      real*8 Tg,T0,C1,C2,AFK
      real*8 aln10,Trel0,Trel1,h0,h1
      parameter(aln10=2.302585093)
      !
      if (cmname(1:12).eq.'ACTIVE_MAT_1') then
        !write(7,*) 'Active Material here'
        Tg = 57.0
        T0 = 17.0
        C1 = 17.44
        C2 = 50.5
        AFK = -24000.0
      else if (cmname(1:12).eq.'ACTIVE_MAT_2') then
        Tg = 38.0
        T0 = -3.0
        C1 = 17.44
        C2 = 42.1
        AFK = -23000.0
      else if (cmname(1:11).eq.'PASSIVE_MAT') then
        !write(7,*) 'Passive Material here'
        Tg = 2.0
        T0 = -11.0
        C1 = 11.44
        C2 = 50.3
        AFK = -20000.0
      else
        write(7,*) 'Big problem this should never happen'
        call xit
      endif
      !
      ! calculate relative temperatures
      !
      Trel0 = temp - dtemp - T0
      Trel1 = Trel0 + dtemp
      !
      ! calculate shift factors
      !
      if (temp.ge.T0) then
        h0 = -aln10*C1*Trel0/(C2 + Trel0)
        h1 = -aln10*C1*Trel1/(C2 + Trel1)
        !
        shift(1) = exp(h0)
        shift(2) = exp(h1)
      else
        h0 = -aln10*AFK*(1/(temp-dtemp + 273.15) - 1/(T0 + 273.15))
        h1 = -aln10*AFK*(1/(temp + 273.15) - 1/(T0 + 273.15))
        !
        shift(1) = exp(h0)
        shift(2) = exp(h1)
      endif
      !
      end subroutine