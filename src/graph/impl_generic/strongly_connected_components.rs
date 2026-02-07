//! Strongly connected components via Tarjan's algorithm (directed graphs).
//!
//! For undirected graphs, equivalent to connected_components.
//! Implemented sequentially at API boundary (DFS-based).

use numr::error::Result;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::graph::traits::types::{ComponentResult, GraphData};

use super::helpers::extract_csr_arrays;

/// Tarjan's strongly connected components algorithm.
///
/// Time: O(V + E). Uses DFS with a stack to identify SCCs.
/// For undirected graphs, equivalent to connected components.
pub fn tarjan_impl<R, C>(_client: &C, graph: &GraphData<R>) -> Result<ComponentResult<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    // Extract CSR at API boundary
    let (row_ptrs, col_indices, _values, n) = extract_csr_arrays(graph)?;

    // Get device from graph
    let device = match &graph.adjacency {
        numr::sparse::SparseTensor::Csr(csr) => csr.values().device().clone(),
        _ => unreachable!(),
    };

    // Build adjacency list for easier traversal
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for u in 0..n {
        let start = row_ptrs[u] as usize;
        let end = row_ptrs[u + 1] as usize;
        for i in start..end {
            let v = col_indices[i] as usize;
            adj[u].push(v);
        }
    }

    // State for Tarjan's algorithm
    let mut index_counter = 0;
    let mut stack: Vec<usize> = Vec::new();
    let mut indices = vec![-1i64; n];
    let mut lowlinks = vec![-1i64; n];
    let mut on_stack = vec![false; n];
    let mut labels = vec![-1i64; n];
    let mut num_components = 0;

    // DFS helper (inline)
    fn tarjan_dfs(
        u: usize,
        index_counter: &mut i64,
        stack: &mut Vec<usize>,
        indices: &mut [i64],
        lowlinks: &mut [i64],
        on_stack: &mut [bool],
        labels: &mut [i64],
        num_components: &mut usize,
        adj: &[Vec<usize>],
    ) {
        indices[u] = *index_counter;
        lowlinks[u] = *index_counter;
        *index_counter += 1;
        stack.push(u);
        on_stack[u] = true;

        for &v in &adj[u] {
            if indices[v] == -1 {
                // Unvisited
                tarjan_dfs(
                    v,
                    index_counter,
                    stack,
                    indices,
                    lowlinks,
                    on_stack,
                    labels,
                    num_components,
                    adj,
                );
                lowlinks[u] = lowlinks[u].min(lowlinks[v]);
            } else if on_stack[v] {
                // Back edge
                lowlinks[u] = lowlinks[u].min(indices[v]);
            }
        }

        // If u is a root node, pop the stack and assign component
        if lowlinks[u] == indices[u] {
            let comp_id = *num_components as i64;
            loop {
                // Safety: u is always on the stack when we reach this point
                let Some(w) = stack.pop() else { break };
                on_stack[w] = false;
                labels[w] = comp_id;
                if w == u {
                    break;
                }
            }
            *num_components += 1;
        }
    }

    // Run Tarjan's from all unvisited nodes
    for u in 0..n {
        if indices[u] == -1 {
            tarjan_dfs(
                u,
                &mut index_counter,
                &mut stack,
                &mut indices,
                &mut lowlinks,
                &mut on_stack,
                &mut labels,
                &mut num_components,
                &adj,
            );
        }
    }

    // Create output tensor
    let labels_tensor = Tensor::<R>::from_slice(&labels, &[n], &device);

    Ok(ComponentResult {
        labels: labels_tensor,
        num_components,
    })
}
