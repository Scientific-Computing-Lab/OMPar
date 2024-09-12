/* Calculate the damage of each node.
 *
 * nlist - An (n, local_size) array containing the neighbour lists,
 *         a value of -1 corresponds to a broken bond.
 * family - An array of the initial number of neighbours for each node.
 * n_neigh - An array of the number of neighbours (particles bound) for each node.
 * damage - An array of the damage for each node. 
 * local_cache - local (local_size) array to store the bond breakages.
 */
void damage_of_node(
  const int n,
  const int *__restrict nlist,
  const int *__restrict family,
        int *__restrict n_neigh,
     double *__restrict damage)
{
  {
    int local_cache[BS];
    {
      const int local_id = omp_get_thread_num();
      const int local_size = BS;
      const int nid = omp_get_team_num();
      const int global_id = nid * local_size + local_id;
      if (global_id < n) {
        //Copy values into local memory 
        local_cache[local_id] = nlist[global_id] != -1 ? 1 : 0; 

        //Wait for all threads

        for (int i = local_size/2; i > 0; i /= 2) {
          if(local_id < i) {
            local_cache[local_id] += local_cache[local_id + i];
          }
          //Wait for all threads
        }

        if (local_id == 0) {
          // Update damage and n_neigh
          int neighbours = local_cache[0];
          n_neigh[nid] = neighbours;
          damage[nid] = 1.0 - (double) neighbours / (double) (family[nid]);
        }
      }
    }
  }
}
