import fsspec

fs = fsspec.filesystem("osdf")

# Ejemplo con un bloque real de O3b (tiempo GPS inicial 1256656896, duración 4096s)
nombre_archivo = "H-H1_GWOSC_O3b_4KHZ_R1-1256656896-4096.hdf5"
ruta_remota = f"/gwdata/O3b/strain.4k/hdf.v1/H1/{nombre_archivo}"

print(f"Descargando {nombre_archivo}...")
fs.get(ruta_remota, nombre_archivo)
print("¡Descarga completada!")

