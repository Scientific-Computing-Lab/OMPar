/*
 * EasyWave - A realtime tsunami simulation program with GPU support.
 * Copyright (C) 2014  Andrey Babeyko, Johannes Spazier
 * GFZ German Research Centre for Geosciences (http://www.gfz-potsdam.de)
 *
 * Parts of this program (especially the GPU extension) were developed
 * within the context of the following publicly funded project:
 * - TRIDEC, EU 7th Framework Programme, Grant Agreement 258723
 *   (http://www.tridec-online.eu)
 *
 * Licensed under the EUPL, Version 1.1 or - as soon they will be approved by
 * the European Commission - subsequent versions of the EUPL (the "Licence"),
 * complemented with the following provision: For the scientific transparency
 * and verification of results obtained and communicated to the public after
 * using a modified version of the work, You (as the recipient of the source
 * code and author of this modified version, used to produce the published
 * results in scientific communications) commit to make this modified source
 * code available in a repository that is easily and freely accessible for a
 * duration of five years after the communication of the obtained results.
 * 
 * You may not use this work except in compliance with the Licence.
 * 
 * You may obtain a copy of the Licence at:
 * https://joinup.ec.europa.eu/software/page/eupl
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and
 * limitations under the Licence.
 */
#include <stdio.h>
#include <string.h>
#include "utilits.h"
#include "cOkadaFault.h"
static const double Rearth = 6384.e+3;
//=========================================================================
// Constructor

cOkadaFault::cOkadaFault()
{
// obligatory user parameters
  (this) -> lat = (this) -> lon = (this) -> depth = (this) -> strike = (this) -> dip = (this) -> rake = 1.e+30;
// optional user parameters
  (this) -> mw = (this) -> slip = (this) -> length = (this) -> width = 0.;
  (this) -> adjust_depth = 0;
// default values
  (this) -> refpos = 0;
  (this) -> mu = 3.5e+10;
// derivative parameters
  (this) -> zbot = (this) -> sind = (this) -> cosd = (this) -> sins = (this) -> coss = (this) -> tand = (this) -> coslat = (this) -> wp = (this) -> dslip = (this) -> sslip = 0;
// flags
  (this) -> checked = 0;
}
//=========================================================================
// Destructor

cOkadaFault::~cOkadaFault()
{
}
//=========================================================================
// read fault parameters from input string. read only. consistency check in separate method

int cOkadaFault::read(char *faultparam)
{
  char *cp;
  char buf[64];
  int ierr;
// origin location
  cp = strstr(faultparam,"-location");
  if (cp) {
    if (sscanf(cp,"%*s %lf %lf %lf",&(this) -> lon,&(this) -> lat,&(this) -> depth) != 3) 
      return Err .  post ("cOkadaFault::read: position");
    (this) -> depth *= 1000;
  }
// adjust depth if fault breaks through surface
  cp = strstr(faultparam,"-adjust_depth");
  if (cp) 
    (this) -> adjust_depth = 1;
// reference point
  cp = strstr(faultparam,"-refpos");
  if (cp) {
    if (sscanf(cp,"%*s %s",buf) != 1) 
      return Err .  post ("cOkadaFault::read: refpos");
    if (!(strcmp(buf,"C")) || !(strcmp(buf,"c"))) 
      (this) -> refpos = 0;
     else if (!(strcmp(buf,"MT")) || !(strcmp(buf,"mt"))) 
      (this) -> refpos = 1;
     else if (!(strcmp(buf,"BT")) || !(strcmp(buf,"bt"))) 
      (this) -> refpos = 2;
     else if (!(strcmp(buf,"BB")) || !(strcmp(buf,"bb"))) 
      (this) -> refpos = 3;
     else if (!(strcmp(buf,"MB")) || !(strcmp(buf,"mb"))) 
      (this) -> refpos = 4;
     else 
      return Err .  post ("cOkadaFault::read: refpos");
  }
// magnitude
  cp = strstr(faultparam,"-mw");
  if (cp) 
    if (sscanf(cp,"%*s %lf",&(this) -> mw) != 1) 
      return Err .  post ("cOkadaFault::read: mw");
// slip
  cp = strstr(faultparam,"-slip");
  if (cp) {
    if (sscanf(cp,"%*s %lf",&(this) -> slip) != 1) 
      return Err .  post ("cOkadaFault::read: slip");
    if ((this) -> slip < 1.e-6) 
      (this) -> slip = 1.e-6;
  }
// strike
  cp = strstr(faultparam,"-strike");
  if (cp) 
    if (sscanf(cp,"%*s %lf",&(this) -> strike) != 1) 
      return Err .  post ("cOkadaFault::read: strike");
// dip
  cp = strstr(faultparam,"-dip");
  if (cp) 
    if (sscanf(cp,"%*s %lf",&(this) -> dip) != 1) 
      return Err .  post ("cOkadaFault::read: dip");
// rake
  cp = strstr(faultparam,"-rake");
  if (cp) 
    if (sscanf(cp,"%*s %lf",&(this) -> rake) != 1) 
      return Err .  post ("cOkadaFault::read: rake");
// length and width
  cp = strstr(faultparam,"-size");
  if (cp) {
    if (sscanf(cp,"%*s %lf %lf",&(this) -> length,&(this) -> width) != 2) 
      return Err .  post ("cOkadaFault::read: size");
    (this) -> length *= 1000;
    (this) -> width *= 1000;
  }
// rigidity
  cp = strstr(faultparam,"-rigidity");
  if (cp) 
    if (sscanf(cp,"%*s %lf",&(this) -> mu) != 1) 
      return Err .  post ("cOkadaFault::read: rigidity");
// check fault data for integrity
//ierr = check(); if(ierr) return ierr;
  return 0;
}
//================================================================================

int cOkadaFault::check()
// Check readed fault parameters for consistency and calculate secondary parameters
{
// check necessary parameters
  if ((this) -> lon == 1.e+30) {
    Err .  post ("cOkadaFault::check: lon");
    return 1;
  }
  if ((this) -> lat == 1.e+30) {
    Err .  post ("cOkadaFault::check: lat");
    return 1;
  }
  if ((this) -> depth == 1.e+30) {
    Err .  post ("cOkadaFault::check: depth");
    return 1;
  }
  if ((this) -> strike == 1.e+30) {
    Err .  post ("cOkadaFault::check: strike");
    return 6;
  }
  if ((this) -> rake == 1.e+30) {
    Err .  post ("cOkadaFault::check: rake");
    return 1;
  }
  if ((this) -> dip == 1.e+30) {
    Err .  post ("cOkadaFault::check: dip");
    return 1;
  }
// cache trigonometric expressions
  (this) -> sind = sin(((double )((this) -> dip)) * 3.14159265358979 / 180);
  (this) -> cosd = cos(((double )((this) -> dip)) * 3.14159265358979 / 180);
  (this) -> sins = sin(((double )(90 - (this) -> strike)) * 3.14159265358979 / 180);
  (this) -> coss = cos(((double )(90 - (this) -> strike)) * 3.14159265358979 / 180);
  (this) -> tand = tan(((double )((this) -> dip)) * 3.14159265358979 / 180);
  (this) -> coslat = cos(((double )((this) -> lat)) * 3.14159265358979 / 180);
// branching through given parameters (the full solution table see end of file)
  if (!((this) -> mw) && !((this) -> slip)) {
    Err .  post ("cOkadaFault::check: not enough data");
    return 1;
  }
   else if (!((this) -> mw) && ((this) -> slip)) {
    if (!((this) -> length) && !((this) -> width)) {
      Err .  post ("cOkadaFault::check: not enough data");
      return 1;
    }
     else if (((this) -> length) && !((this) -> width)) {
      (this) -> width = (this) -> length / 2;
    }
     else if (!((this) -> length) && ((this) -> width)) {
      (this) -> length = 2 * (this) -> width;
    }
     else if (((this) -> length) && ((this) -> width)) {
    }
     else {
      Err .  post ("cOkadaFault::check: internal error");
      return 4;
    }
    (this) -> mw = 2. / 3. * (log10((this) -> mu * (this) -> length * (this) -> width * (this) -> slip) - 9.1);
  }
   else if (((this) -> mw) && !((this) -> slip)) {
    if (!((this) -> length) && !((this) -> width)) {
// scaling relations used by JMA
      (this) -> length = pow(10.,- 1.80 + 0.5 * (this) -> mw) * 1000;
      (this) -> width = (this) -> length / 2;
    }
     else if (((this) -> length) && !((this) -> width)) {
      (this) -> width = (this) -> length / 2;
    }
     else if (!((this) -> length) && ((this) -> width)) {
      (this) -> length = 2 * (this) -> width;
    }
     else if (((this) -> length) && ((this) -> width)) {
    }
     else {
      Err .  post ("cOkadaFault::check: internal error");
      return 4;
    }
    (this) -> slip = (this) ->  mw2m0 () / (this) -> mu / (this) -> length / (this) -> width;
  }
   else if (((this) -> mw) && ((this) -> slip)) {
    if (!((this) -> length) && !((this) -> width)) {
      double area = (this) ->  mw2m0 () / (this) -> mu / (this) -> slip;
      (this) -> length = sqrt(2 * area);
      (this) -> width = (this) -> length / 2;
    }
     else if (((this) -> length) && !((this) -> width)) {
      (this) -> width = (this) ->  mw2m0 () / (this) -> mu / (this) -> slip / (this) -> length;
    }
     else if (!((this) -> length) && ((this) -> width)) {
      (this) -> length = (this) ->  mw2m0 () / (this) -> mu / (this) -> slip / (this) -> width;
    }
     else if (((this) -> length) && ((this) -> width)) {
      if (fabs(1 - (this) -> mu * (this) -> slip * (this) -> length * (this) -> width / (this) ->  mw2m0 ()) > 0.01) {
        Err .  post ("cOkadaFault::check: data inconsistency");
        return 1;
      }
    }
     else {
      Err .  post ("cOkadaFault::check: internal error");
      return 4;
    }
  }
// calculate bottom of the fault
  switch((this) -> refpos){
    double ztop;
    case 0:
    ztop = (this) -> depth - (this) -> width / 2 * (this) -> sind;
    if (ztop < 0) {
      if (((this) -> adjust_depth)) {
        ztop = 0.;
        (this) -> depth = ztop + (this) -> width / 2 * (this) -> sind;
      }
       else {
        Err .  post ("cOkadaFault::check: negative ztop");
        return 3;
      }
    }
    (this) -> zbot = (this) -> depth + (this) -> width / 2 * (this) -> sind;
    break; 
    case 1:;
    case 2:
    (this) -> zbot = (this) -> depth + (this) -> width * (this) -> sind;
    break; 
    case 3:;
    case 4:
    ztop = (this) -> depth - (this) -> width * (this) -> sind;
    if (ztop < 0) {
      if (((this) -> adjust_depth)) {
        ztop = 0.;
        (this) -> depth = ztop + (this) -> width * (this) -> sind;
      }
       else {
        Err .  post ("cOkadaFault::check: negative ztop");
        return 3;
      }
    }
    (this) -> zbot = (this) -> depth;
    break; 
  }
// slip components
  (this) -> dslip = (this) -> slip * sin(((double )((this) -> rake)) * 3.14159265358979 / 180);
  (this) -> sslip = (this) -> slip * cos(((double )((this) -> rake)) * 3.14159265358979 / 180);
// surface projection of width
  (this) -> wp = (this) -> width * (this) -> cosd;
  (this) -> checked = 1;
  return 0;
}
//=========================================================================

double cOkadaFault::mw2m0()
{
  return pow(10.,3. * (this) -> mw / 2 + 9.1);
}
//=========================================================================

double cOkadaFault::getM0()
{
  if (!((this) -> checked)) {
    Err .  post ("cOkadaFault::getM0: fault not checked");
    return - 1.e+30;
  }
  return (this) ->  mw2m0 ();
}
//=========================================================================

double cOkadaFault::getMw()
{
  if (!((this) -> checked)) {
    Err .  post ("cOkadaFault::getMw: fault not checked");
    return - 1.e+30;
  }
  return (this) -> mw;
}
//=========================================================================

double cOkadaFault::getZtop()
{
  return (this) -> zbot - (this) -> width * (this) -> sind;
}
//=========================================================================

int cOkadaFault::global2local(double glon,double glat,double &lx,double &ly)
// from global geographical coordinates into local Okada's coordinates
{
  double x;
  double y;
// center coordinate system to refpos (lon/lat), convert to meters
  y = Rearth * (((double )(glat - (this) -> lat)) * 3.14159265358979 / 180);
  x = Rearth * (this) -> coslat * (((double )(glon - (this) -> lon)) * 3.14159265358979 / 180);
// rotate according to strike
  lx = x * (this) -> coss + y * (this) -> sins;
  ly = -x * (this) -> sins + y * (this) -> coss;
// finally shift to Okada's origin point (BB)
  switch((this) -> refpos){
    case 0:
    lx = lx + (this) -> length / 2;
    ly = ly + (this) -> wp / 2;
    break; 
    case 1:
    lx = lx + (this) -> length / 2;
    ly = ly + (this) -> wp;
    break; 
    case 2:
    lx = lx;
    ly = ly + (this) -> wp;
    break; 
    case 3:
    lx = lx;
    ly = ly;
    break; 
    case 4:
    lx = lx + (this) -> length / 2;
    ly = ly;
    break; 
  }
  return 0;
}
//=========================================================================

int cOkadaFault::local2global(double lx,double ly,double &glon,double &glat)
// from local Okada's coordinates to global geographical
{
  double x;
  double y;
  double gx;
  double gy;
// define local coordinates relative to the fault refpos
  switch((this) -> refpos){
    case 0:
    x = lx - (this) -> length / 2;
    y = ly - (this) -> wp / 2;
    break; 
    case 1:
    x = lx - (this) -> length / 2;
    y = ly - (this) -> wp;
    break; 
    case 2:
    x = lx;
    y = ly - (this) -> wp;
    break; 
    case 3:
    x = lx;
    y = ly;
    break; 
    case 4:
    x = lx - (this) -> length / 2;
    y = ly;
    break; 
  }
// back-rotate to geographical axes (negative strike!). Values are still in meters!
  gx = x * (this) -> coss + y * -(this) -> sins;
  gy = -x * -(this) -> sins + y * (this) -> coss;
// convert meters to degrees. This is offset in degrees relative to refpos. Add refpos coordinates for absolute values
  glat = ((double )(gy / Rearth)) / 3.14159265358979 * 180 + (this) -> lat;
  glon = ((double )(gx / Rearth / cos(((double )((this) -> lat)) * 3.14159265358979 / 180))) / 3.14159265358979 * 180 + (this) -> lon;
  return 0;
}
//=========================================================================
// get ruptured area to calculate surface displacement
// parameter rand accounts for lateral enlargement at rands of rupture surface projection

int cOkadaFault::getDeformArea(double &lonmin,double &lonmax,double &latmin,double &latmax)
{
  #define FLT_NL 2    // significant deformation area along length (in length units)
  #define FLT_NW 5    // significant deformation area along width (in width units)
  int ierr;
  double dxC;
  double dyC;
  double l2;
  double w2;
  double glon;
  double glat;
  if (!((this) -> checked)) {
    Err .  post ("cOkadaFault::getDeformArea: attempt with non-checked fault");
    return 4;
  }
// follow rectangle around the fault
  dxC = 2 * (this) -> length;
  l2 = (this) -> length / 2;
  dyC = 5 * (this) -> wp;
  w2 = (this) -> wp / 2;
  (this) ->  local2global ((dxC + l2),(dyC + w2),glon,glat);
  lonmin = lonmax = glon;
  latmin = latmax = glat;
  (this) ->  local2global ((-dxC + l2),(dyC + w2),glon,glat);
  if (glon < lonmin) 
    lonmin = glon;
  if (glon > lonmax) 
    lonmax = glon;
  if (glat < latmin) 
    latmin = glat;
  if (glat > latmax) 
    latmax = glat;
  (this) ->  local2global ((-dxC + l2),(-dyC + w2),glon,glat);
  if (glon < lonmin) 
    lonmin = glon;
  if (glon > lonmax) 
    lonmax = glon;
  if (glat < latmin) 
    latmin = glat;
  if (glat > latmax) 
    latmax = glat;
  (this) ->  local2global ((dxC + l2),(-dyC + w2),glon,glat);
  if (glon < lonmin) 
    lonmin = glon;
  if (glon > lonmax) 
    lonmax = glon;
  if (glat < latmin) 
    latmin = glat;
  if (glat > latmax) 
    latmax = glat;
  return 0;
}
//=========================================================================

int cOkadaFault::calculate(double lon0,double lat0,double &uz,double &ulon,double &ulat)
{
  int ierr;
  double x;
  double y;
  double ux;
  double uy;
  if (!((this) -> checked)) {
    Err .  post ("cOkadaFault::calculate: attempt with non-checked fault");
    return 4;
  }
  (this) ->  global2local (lon0,lat0,x,y);
// Okada model
  okada((this) -> length,(this) -> width,(this) -> zbot,(this) -> sind,(this) -> cosd,(this) -> sslip,(this) -> dslip,x,y,1,&ux,&uy,&uz);
// back-rotate horizontal deformations to global coordinates (negative strike!)
  ulon = ux * (this) -> coss + uy * -(this) -> sins;
  ulat = -ux * -(this) -> sins + uy * (this) -> coss;
  return 0;
}
//=========================================================================

int cOkadaFault::calculate(double lon0,double lat0,double &uz)
{
  int ierr;
  double x;
  double y;
  double ux;
  double uy;
  if (!((this) -> checked)) {
    Err .  post ("cOkadaFault::calculate: attempt with non-checked fault");
    return 4;
  }
  (this) ->  global2local (lon0,lat0,x,y);
// Okada model
  okada((this) -> length,(this) -> width,(this) -> zbot,(this) -> sind,(this) -> cosd,(this) -> sslip,(this) -> dslip,x,y,0,&ux,&uy,&uz);
  return 0;
}
//======================================================================================
// Input parameter selection: Solution table
// if given:
// mw slip L W  action
// -  -  -  -  err_data
// -  -  -  +  err_data
// -  -  +  -  err_data
// -  -  +  +  err_data
// -  +  -  -  err_data
// -  +  -  +  L=2W, Mw
// -  +  +  -  W=L/2, Mw
// -  +  +  +  Mw
// +  -  -  -  W&C96(L,W), S
// +  -  -  +  L=2W, S
// +  -  +  -  W=L/2, S
// +  -  +  +  S
// +  +  -  -  area(Mw), L=2W
// +  +  -  +  L
// +  +  +  -  W
// +  +  +  +  check Mw=muSLW
