CREATE DATABASE  ColorDetector;
go

use ColorDetector;
go

CREATE TABLE fotos (
	id_foto INT IDENTITY(1,1) PRIMARY KEY,
	nombre_archivo NVARCHAR(255) NOT NULL,
	fecha_hora DATETIME DEFAULT GETDATE(),
	ancho INT,
	alto INT,
	total_pixeles INT
)
go

CREATE TABLE objetos_azul (
	id_objeto INT IDENTITY(1,1) PRIMARY KEY,
	id_foto INT FOREIGN KEY REFERENCES fotos(id_foto),
	x INT NOT NULL,
	y INT NOT NULL,
	area FLOAT NOT NULL,
	fecha_deteccion DATETIME DEFAULT GETDATE()
)
go

CREATE TABLE objetos_verde (
	id_objeto INT IDENTITY(1,1) PRIMARY KEY,
	id_foto INT FOREIGN KEY REFERENCES fotos(id_foto),
	x INT NOT NULL,
	y INT NOT NULL,
	area FLOAT NOT NULL,
	fecha_deteccion DATETIME DEFAULT GETDATE()
)
go

CREATE TABLE objetos_rojo (
	id_objeto INT IDENTITY(1,1) PRIMARY KEY,
	id_foto INT FOREIGN KEY REFERENCES fotos(id_foto),
	x INT NOT NULL,
	y INT NOT NULL,
	area FLOAT NOT NULL,
	fecha_deteccion DATETIME DEFAULT GETDATE()
)
go

CREATE TABLE objetos_amarillo (
	id_objeto INT IDENTITY(1,1) PRIMARY KEY,
	id_foto INT FOREIGN KEY REFERENCES fotos(id_foto),
	x INT NOT NULL,
	y INT NOT NULL,
	area FLOAT NOT NULL,
	fecha_deteccion DATETIME DEFAULT GETDATE()
)
go

-- Agregar columna para la matriz de la imagen (almacenada como JSON o texto)
ALTER TABLE fotos ADD matriz_imagen NVARCHAR(MAX) NULL;
GO

-- Verificar que se agregó la columna
SELECT COLUMN_NAME, DATA_TYPE 
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_NAME = 'fotos' AND TABLE_CATALOG = 'ColorDetector';

SELECT 
    id_foto,
    nombre_archivo,
    ancho,
    alto,
    total_pixeles,
    LEN(matriz_imagen) as longitud_matriz,
    LEFT(matriz_imagen, 100) as preview_matriz  -- Primeros 100 caracteres
FROM fotos
WHERE matriz_imagen IS NOT NULL;

SELECT 
    COLUMN_NAME, 
    DATA_TYPE, 
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_NAME = 'fotos' 
ORDER BY ORDINAL_POSITION;


SELECT * FROM fotos
SELECT * FROM objetos_amarillo
SELECT * FROM objetos_azul
SELECT * FROM objetos_rojo
SELECT * FROM objetos_verde