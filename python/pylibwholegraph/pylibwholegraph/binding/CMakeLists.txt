# Set the list of Cython files to build
set(cython_sources wholememory_binding.pyx)
set(linked_libraries wholegraph::wholegraph)

# Build all of the Cython targets
rapids_cython_create_modules(
        CXX
        SOURCE_FILES "${cython_sources}"
        LINKED_LIBRARIES "${linked_libraries}"
        ASSOCIATED_TARGETS wholegraph
        MODULE_PREFIX wholegraphcif)
