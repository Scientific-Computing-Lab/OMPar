
int Quantities_scalefactor_space_acceldir(int ix_g, int iy_g, int iz_g)
{
  int result = 0;

#ifndef RELAXED_TESTING
  const int im = 134456;
  const int ia = 8121;
  const int ic = 28411;

  result = ( (result+(ix_g+2))*ia + ic ) % im;
  result = ( (result+(iy_g+2))*ia + ic ) % im;
  result = ( (result+(iz_g+2))*ia + ic ) % im;
  result = ( (result+(ix_g+3*iy_g+7*iz_g+2))*ia + ic ) % im;
  result = ix_g+3*iy_g+7*iz_g+2;
  result = result & ( (1<<2) - 1 );
#endif
  result = 1 << result;

  return result;
}


P Quantities_init_face_acceldir(int ia, int ie, int iu, int scalefactor_space, int octant)
{
  /*--- Quantities_affinefunction_ inline ---*/
  return ( (P) (1 + ia) ) 

    /*--- Quantities_scalefactor_angle_ inline ---*/
    * ( (P) (1 << (ia & ( (1<<3) - 1))) ) 

    /*--- Quantities_scalefactor_space_ inline ---*/
    * ( (P) scalefactor_space)

    /*--- Quantities_scalefactor_energy_ inline ---*/
    * ( (P) (1 << ((( (ie) * 1366 + 150889) % 714025) & ( (1<<2) - 1))) )

    /*--- Quantities_scalefactor_unknown_ inline ---*/
    * ( (P) (1 << ((( (iu) * 741 + 60037) % 312500) & ( (1<<2) - 1))) )

    /*--- Quantities_scalefactor_octant_ ---*/
    * ( (P) 1 + octant);
}

void Quantities_solve_acceldir(P* __restrict vs_local,
                               Dimensions dims,
                               P*__restrict facexy,
                               P*__restrict facexz,
                               P*__restrict faceyz,
                               int ix, int iy, int iz,
                               int ix_g, int iy_g, int iz_g,
                               int ie, int ia,
                               int octant, int octant_in_block, int noctant_per_block)
{
  const int dir_x = Dir_x( octant );
  const int dir_y = Dir_y( octant );
  const int dir_z = Dir_z( octant );

  int iu = 0;

  /*---Average the face values and accumulate---*/

  /*---The state value and incoming face values are first adjusted to
    normalized values by removing the spatial scaling.
    They are then combined using a weighted average chosen in a special
    way to give just the expected result.
    Finally, spatial scaling is applied to the result which is then
    stored.
    ---*/

  /*--- Quantities_scalefactor_octant_ inline ---*/
  const P scalefactor_octant = 1 + octant;
  const P scalefactor_octant_r = ((P)1) / scalefactor_octant;

  /*---Quantities_scalefactor_space_ inline ---*/
  const P scalefactor_space = (P)Quantities_scalefactor_space_acceldir(ix_g, iy_g, iz_g);
  const P scalefactor_space_r = ((P)1) / scalefactor_space;
  const P scalefactor_space_x_r = ((P)1) /
    Quantities_scalefactor_space_acceldir( ix_g - dir_x, iy_g, iz_g );
  const P scalefactor_space_y_r = ((P)1) /
    Quantities_scalefactor_space_acceldir( ix_g, iy_g - dir_y, iz_g );
  const P scalefactor_space_z_r = ((P)1) /
    Quantities_scalefactor_space_acceldir( ix_g, iy_g, iz_g - dir_z );

  for( iu=0; iu<NU; ++iu )
    {

      int vs_local_index = ia + dims.na * (
                           iu + NU  * (
                           ie + dims.ne * (
                           ix + dims.ncell_x * (
                           iy + dims.ncell_y * (
                           octant + NOCTANT * (
                           0))))));

      const P result = ( vs_local[vs_local_index] * scalefactor_space_r + 
               (
                /*--- ref_facexy inline ---*/
                facexy[ia + dims.na      * (
                        iu + NU           * (
                        ie + dims.ne      * (
                        ix + dims.ncell_x * (
                        iy + dims.ncell_y * (
                        octant + NOCTANT * (
                        0 )))))) ]

               /*--- Quantities_xfluxweight_ inline ---*/
               * (P) ( 1 / (P) 2 )

               * scalefactor_space_z_r

               /*--- ref_facexz inline ---*/
               + facexz[ia + dims.na      * (
                        iu + NU           * (
                        ie + dims.ne      * (
                        ix + dims.ncell_x * (
                        iz + dims.ncell_z * (
                        octant + NOCTANT * (
                        0 )))))) ]

               /*--- Quantities_yfluxweight_ inline ---*/
               * (P) ( 1 / (P) 4 )

               * scalefactor_space_y_r

               /*--- ref_faceyz inline ---*/
               + faceyz[ia + dims.na      * (
                        iu + NU           * (
                        ie + dims.ne      * (
                        iy + dims.ncell_y * (
                        iz + dims.ncell_z * (
                        octant + NOCTANT * (
                        0 )))))) ]

                        /*--- Quantities_zfluxweight_ inline ---*/
                        * (P) ( 1 / (P) 4 - 1 / (P) (1 << ( ia & ( (1<<3) - 1 ) )) )

               * scalefactor_space_x_r
               ) 
               * scalefactor_octant_r ) * scalefactor_space;

      vs_local[vs_local_index] = result;

      const P result_scaled = result * scalefactor_octant;
      /*--- ref_facexy inline ---*/
      facexy[ia + dims.na      * (
             iu + NU           * (
             ie + dims.ne      * (
             ix + dims.ncell_x * (
             iy + dims.ncell_y * (
             octant + NOCTANT * (
             0 )))))) ] = result_scaled;

      /*--- ref_facexz inline ---*/
      facexz[ia + dims.na      * (
             iu + NU           * (
             ie + dims.ne      * (
             ix + dims.ncell_x * (
             iz + dims.ncell_z * (
             octant + NOCTANT * (
             0 )))))) ] = result_scaled;

      /*--- ref_faceyz inline ---*/
      faceyz[ia + dims.na      * (
             iu + NU           * (
             ie + dims.ne      * (
             iy + dims.ncell_y * (
             iz + dims.ncell_z * (
             octant + NOCTANT * (
             0 )))))) ] = result_scaled;

    } /*---for---*/
}


void Sweeper_sweep_cell_acceldir( const Dimensions &dims,
                                  int wavefront,
                                  int octant,
                                  int ix, int iy,
                                  int ix_g, int iy_g, int iz_g,
                                  int dir_x, int dir_y, int dir_z,
                                  P* __restrict facexy,
                                  P* __restrict facexz,
                                  P* __restrict faceyz,
                                  const P* __restrict a_from_m,
                                  const P* __restrict m_from_a,
                                  const P* __restrict vi,
                                  P* __restrict vo,
                                  P* __restrict vs_local,
                                  int octant_in_block,
                                  int noctant_per_block,
                                  int ie)
{
  /*---Declarations---*/
  int im = 0;
  int ia = 0;
  int iu = 0;
  /* int octant = 0; */

  /*--- Dimensions ---*/
  int dims_ncell_x = dims.ncell_x;
  int dims_ncell_y = dims.ncell_y;
  int dims_ncell_z = dims.ncell_z;
  int dims_ne = dims.ne;
  int dims_na = dims.na;
  int dims_nm = dims.nm;

  /*--- Solve for Z dimension, and check bounds.
    The sum of the dimensions should equal the wavefront number.
    If z < 0 or z > wavefront number, we are out of bounds.
    Z also shouldn't exceed the spacial bound for the z dimension.

    The calculation is adjusted for the direction of each axis
    in a given octant.
  ---*/

  const int ixwav = dir_x==DIR_UP ? ix : (dims_ncell_x-1) - ix;
  const int iywav = dir_y==DIR_UP ? iy : (dims_ncell_y-1) - iy;
  const int izwav = wavefront - ixwav - iywav;
  const int iz = dir_z==DIR_UP ? izwav : (dims_ncell_z-1) - izwav;

//  int ixwav, iywav, izwav;
//  if (dir_x==DIR_UP) { ixwav = ix; } else { ixwav = (dims_ncell_x-1) - ix; }
//  if (dir_y==DIR_UP) { iywav = iy; } else { iywav = (dims_ncell_y-1) - iy; }
  
//  if (dir_z==DIR_UP) {
//    iz = wavefront - (ixwav + iywav); } 
//  else { 
//    iz = (dims_ncell_z-1) - (wavefront - (ixwav + iywav));
//  }

  /*--- Bounds check ---*/
  if ((iz >= 0 && iz < dims_ncell_z) )// &&
    /* ((dir_z==DIR_UP && iz <= wavefront) || */
    /*  (dir_z==DIR_DN && (dims_ncell_z-1-iz) <= wavefront))) */
    {

   /*---Loop over energy groups---*/
//      for( ie=0; ie<dims_ne; ++ie )
      {

      /*--------------------*/
      /*---Transform state vector from moments to angles---*/
      /*--------------------*/

      /*---This loads values from the input state vector,
           does the small dense matrix-vector product,
           and stores the result in a relatively small local
           array that is hopefully small enough to fit into
           processor cache.
      ---*/

      for( iu=0; iu<NU; ++iu )
      for( ia=0; ia<dims_na; ++ia )
      { 
        // reset reduction
        P result = (P)0;
        for( im=0; im < dims_nm; ++im )
        {
          /*--- const_ref_a_from_m inline ---*/
          result += a_from_m[ ia     + dims_na * (
                              im     +      NM * (
                              octant + NOCTANT * (
                              0 ))) ] * 

            /*--- const_ref_state inline ---*/
            vi[im + dims.nm      * (
                          iu + NU           * (
                          ix + dims_ncell_x * (
                          iy + dims_ncell_y * (
                          ie + dims_ne      * (
                          iz + dims_ncell_z * ( /*---NOTE: This axis MUST be slowest-varying---*/
                          0 ))))))];
        }

        /*--- ref_vslocal inline ---*/
        vs_local[ ia + dims.na * (
                  iu + NU  * (
                  ie + dims_ne * (
                  ix + dims_ncell_x * (
                  iy + dims_ncell_y * (
                  octant + NOCTANT * (
                                       0)))))) ] = result;
      }
      }

      /*--------------------*/
      /*---Perform solve---*/
      /*--------------------*/

//   /*---Loop over energy groups---*/
      for( ia=0; ia<dims_na; ++ia )
      {
        Quantities_solve_acceldir(vs_local, dims, facexy, facexz, faceyz, 
                             ix, iy, iz,
                             ix_g, iy_g, iz_g,
                             ie, ia,
                             octant, octant_in_block, noctant_per_block);
      }

      /*--------------------*/
      /*---Transform state vector from angles to moments---*/
      /*--------------------*/

      /*---Perform small dense matrix-vector products and store
           the result in the output state vector.
      ---*/

   /*---Loop over energy groups---*/
      for( iu=0; iu<NU; ++iu )
      for( im=0; im<dims_nm; ++im )
      {
        P result = (P)0;
        for( ia=0; ia<dims_na; ++ia )
        {
         /*--- const_ref_m_from_a ---*/
         result += m_from_a[ im     +      NM * (
                             ia     + dims_na * (
                             octant + NOCTANT * (
                             0 ))) ] *

         /*--- const_ref_vslocal ---*/
         vs_local[ ia + dims_na * (
                   iu + NU    * (
                   ie + dims_ne * (
                   ix + dims_ncell_x * (
                   iy + dims_ncell_y * (
                   octant + NOCTANT * (
                   0 )))))) ];
        }

        /*--- ref_state inline ---*/
         vo[im + dims.nm     * (
           iu + NU           * (
           ix + dims_ncell_x * (
           iy + dims_ncell_y * (
           ie + dims_ne      * (
           iz + dims_ncell_z * (
           0 ))))))] += result;
      }

//      } /*---ie---*/

    } /*--- iz ---*/
}

