import cv2
import numpy as np
import base64
import os
import json
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import pyodbc

app = Flask(__name__)

DB_CONFIG = {
    'server': 'localhost',
    'database': 'ColorDetector',
    'trusted_connection': 'yes',
    'driver': 'SQL Server'
}

def get_db_connection():
    try:
        conn_str = f"DRIVER={DB_CONFIG['driver']};SERVER={DB_CONFIG['server']};DATABASE={DB_CONFIG['database']};Trusted_Connection=yes;"
        conn = pyodbc.connect(conn_str)
        print("Conexión a la base de datos exitosa")
        return conn
    except Exception as e:
        print(f"Error de conexión a la base de datos: {e}")
        return None

# camera = cv2.VideoCapture('http://192.168.137.192:8080/video')
camera = cv2.VideoCapture(0)

def generar_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_procesado = procesar_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame_procesado)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def procesar_frame(frame):
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    azulBajo = np.array([100, 100, 20], np.uint8)
    azulAlto = np.array([125, 255, 255], np.uint8)
    verdeBajo = np.array([45, 100, 20], np.uint8)
    verdeAlto = np.array([75, 255, 255], np.uint8)
    redBajo1 = np.array([0, 100, 20], np.uint8)
    redAlto1 = np.array([5, 255, 255], np.uint8)
    redBajo2 = np.array([175, 100, 20], np.uint8)
    redAlto2 = np.array([179, 255, 255], np.uint8)
    amarilloBajo = np.array([15, 100, 20], np.uint8)
    amarilloAlto = np.array([45, 255, 255], np.uint8)
    
    maskAzul = cv2.inRange(frameHSV, azulBajo, azulAlto)
    maskVerde = cv2.inRange(frameHSV, verdeBajo, verdeAlto)
    maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
    maskRed2 = cv2.inRange(frameHSV, redBajo2, redAlto2)
    maskRed = cv2.add(maskRed1, maskRed2)
    maskAmarillo = cv2.inRange(frameHSV, amarilloBajo, amarilloAlto)
    
    frame = dibujar_contornos(maskAzul, frame, (255, 0, 0))
    frame = dibujar_contornos(maskVerde, frame, (0, 255, 0))
    frame = dibujar_contornos(maskRed, frame, (0, 0, 255))
    frame = dibujar_contornos(maskAmarillo, frame, (0, 255, 255))
    
    return frame

def dibujar_contornos(mask, frame, color):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 3000:
            M = cv2.moments(c)
            if M["m00"] == 0: 
                M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M['m01'] / M['m00'])
            nuevoContorno = cv2.convexHull(c)
            cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y), font, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.drawContours(frame, [nuevoContorno], 0, color, 3)
    
    return frame

def detectar_objetos(frame):
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    azulBajo = np.array([100, 100, 20], np.uint8)
    azulAlto = np.array([125, 255, 255], np.uint8)
    verdeBajo = np.array([45, 100, 20], np.uint8)
    verdeAlto = np.array([75, 255, 255], np.uint8)
    redBajo1 = np.array([0, 100, 20], np.uint8)
    redAlto1 = np.array([5, 255, 255], np.uint8)
    redBajo2 = np.array([175, 100, 20], np.uint8)
    redAlto2 = np.array([179, 255, 255], np.uint8)
    amarilloBajo = np.array([15, 100, 20], np.uint8)
    amarilloAlto = np.array([45, 255, 255], np.uint8)
    
    maskAzul = cv2.inRange(frameHSV, azulBajo, azulAlto)
    maskVerde = cv2.inRange(frameHSV, verdeBajo, verdeAlto)
    maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
    maskRed2 = cv2.inRange(frameHSV, redBajo2, redAlto2)
    maskRed = cv2.add(maskRed1, maskRed2)
    maskAmarillo = cv2.inRange(frameHSV, amarilloBajo, amarilloAlto)
    
    objetos = {
        'azul': procesar_objetos(maskAzul),
        'verde': procesar_objetos(maskVerde),
        'rojo': procesar_objetos(maskRed),
        'amarillo': procesar_objetos(maskAmarillo)
    }
    
    return objetos

def procesar_objetos(mask):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objetos = []
    
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 3000:
            M = cv2.moments(c)
            if M["m00"] == 0: 
                M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M['m01'] / M['m00'])
            objetos.append({
                'x': x,
                'y': y,
                'area': area
            })
    
    return objetos

def guardar_en_base_datos(nombre_archivo, ancho, alto, total_pixeles, matriz_imagen, objetos):
    try:
        conn = get_db_connection()
        if conn is None:
            return False, "No se pudo conectar a la base de datos"
            
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO fotos (nombre_archivo, ancho, alto, total_pixeles, matriz_imagen) OUTPUT INSERTED.id_foto VALUES (?, ?, ?, ?, ?)",
            nombre_archivo, ancho, alto, total_pixeles, matriz_imagen
        )
        id_foto = cursor.fetchone()[0]
        
        for obj in objetos['azul']:
            cursor.execute(
                "INSERT INTO objetos_azul (id_foto, x, y, area) VALUES (?, ?, ?, ?)",
                id_foto, obj['x'], obj['y'], obj['area']
            )
        
        for obj in objetos['verde']:
            cursor.execute(
                "INSERT INTO objetos_verde (id_foto, x, y, area) VALUES (?, ?, ?, ?)",
                id_foto, obj['x'], obj['y'], obj['area']
            )
        
        for obj in objetos['rojo']:
            cursor.execute(
                "INSERT INTO objetos_rojo (id_foto, x, y, area) VALUES (?, ?, ?, ?)",
                id_foto, obj['x'], obj['y'], obj['area']
            )
        
        for obj in objetos['amarillo']:
            cursor.execute(
                "INSERT INTO objetos_amarillo (id_foto, x, y, area) VALUES (?, ?, ?, ?)",
                id_foto, obj['x'], obj['y'], obj['area']
            )
        
        conn.commit()
        conn.close()
        return True, id_foto
    except Exception as e:
        return False, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_and_save', methods=['POST'])
def capture_and_save():
    try:
        data = request.get_json()
        image_data = data['image_data']
        
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        
        nparr = np.frombuffer(binary_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        filename = f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join('saved_photos', filename)
        cv2.imwrite(filepath, frame)
        
        objetos = detectar_objetos(frame)
        
        # matriz_reducida = frame[::10, ::10]  # Tomar cada 10px para reducir tamaño
        # matriz_json = json.dumps({
        #     'shape': frame.shape,
        #     'sample': matriz_reducida.tolist(),
        #     'total_pixels': frame.shape[0] * frame.shape[1]
        # })
        
        matriz_completa = frame.tolist()
        matriz_json = json.dumps(matriz_completa)
        
        height, width = frame.shape[:2]
        total_pixeles = height * width
        
        exito, resultado = guardar_en_base_datos(
            filename, width, height, total_pixeles, matriz_json, objetos
        )
        
        if exito:
            return jsonify({
                'success': True,
                'filename': filename,
                'detections': {
                    'azul': len(objetos['azul']),
                    'verde': len(objetos['verde']),
                    'rojo': len(objetos['rojo']),
                    'amarillo': len(objetos['amarillo'])
                },
                'id_foto': resultado
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Error en base de datos: {resultado}'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/photos/<filename>')
def serve_photo(filename):
    return send_from_directory('saved_photos', filename)

@app.route('/get_database_stats')
def get_database_stats():
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'success': False, 'error': 'No se pudo conectar a la base de datos'})
            
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM fotos")
        total_fotos = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM objetos_azul")
        objetos_azul = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM objetos_verde")
        objetos_verde = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM objetos_rojo")
        objetos_rojo = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM objetos_amarillo")
        objetos_amarillo = cursor.fetchone()[0]
        
        total_objetos = objetos_azul + objetos_verde + objetos_rojo + objetos_amarillo
        
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_fotos': total_fotos,
                'objetos_azul': objetos_azul,
                'objetos_verde': objetos_verde,
                'objetos_rojo': objetos_rojo,
                'objetos_amarillo': objetos_amarillo,
                'total_objetos': total_objetos
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/view_database')
def view_database():
    return render_template('database.html')

@app.route('/get_db_recent_photos')
def get_db_recent_photos():
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'success': False, 'error': 'No se pudo conectar a la base de datos'})
            
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id_foto, nombre_archivo, fecha_hora, ancho, alto 
            FROM fotos 
            ORDER BY fecha_hora DESC
        """)
        
        fotos = []
        for row in cursor.fetchall():
            fotos.append({
                'id': row[0],
                'nombre': row[1],
                'fecha': row[2].strftime('%Y-%m-%d %H:%M:%S'),
                'ancho': row[3],
                'alto': row[4]
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'fotos': fotos
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_db_objects/<color>')
def get_db_objects(color):
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'success': False, 'error': 'No se pudo conectar a la base de datos'})
            
        cursor = conn.cursor()
        
        tabla = f"objetos_{color}"
        cursor.execute(f"""
            SELECT id_objeto, id_foto, x, y, area, fecha_deteccion 
            FROM {tabla} 
            ORDER BY fecha_deteccion DESC
        """)
        
        objetos = []
        for row in cursor.fetchall():
            objetos.append({
                'id_objeto': row[0],
                'id_foto': row[1],
                'x': row[2],
                'y': row[3],
                'area': row[4],
                'fecha_deteccion': row[5].strftime('%Y-%m-%d %H:%M:%S') if row[5] else 'N/A'
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'objetos': objetos
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_db_all_objects')
def get_db_all_objects():
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'success': False, 'error': 'No se pudo conectar a la base de datos'})
            
        cursor = conn.cursor()
        
        colores = ['azul', 'verde', 'rojo', 'amarillo']
        todos_objetos = []
        
        for color in colores:
            cursor.execute(f"""
                SELECT id_objeto, id_foto, x, y, area, fecha_deteccion 
                FROM objetos_{color} 
                ORDER BY fecha_deteccion DESC
            """)
            
            for row in cursor.fetchall():
                todos_objetos.append({
                    'id_objeto': row[0],
                    'id_foto': row[1],
                    'x': row[2],
                    'y': row[3],
                    'area': row[4],
                    'fecha_deteccion': row[5].strftime('%Y-%m-%d %H:%M:%S') if row[5] else 'N/A',
                    'color': color
                })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'objetos': todos_objetos
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_db_photo_objects/<int:foto_id>')
def get_db_photo_objects(foto_id):
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'success': False, 'error': 'No se pudo conectar a la base de datos'})
            
        cursor = conn.cursor()
        
        objetos = {
            'azul': 0,
            'verde': 0,
            'rojo': 0,
            'amarillo': 0,
            'total': 0
        }
        
        for color in ['azul', 'verde', 'rojo', 'amarillo']:
            cursor.execute(f"SELECT COUNT(*) FROM objetos_{color} WHERE id_foto = ?", foto_id)
            count = cursor.fetchone()[0]
            objetos[color] = count
            objetos['total'] += count
        
        conn.close()
        
        return jsonify({
            'success': True,
            'objetos': objetos
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_db_photo_details/<int:foto_id>')
def get_db_photo_details(foto_id):
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'success': False, 'error': 'No se pudo conectar a la base de datos'})
            
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM fotos WHERE id_foto = ?", foto_id)
        row = cursor.fetchone()
        
        if not row:
            return jsonify({'success': False, 'error': 'Foto no encontrada'})
        
        foto = {
            'id_foto': row[0],
            'nombre_archivo': row[1],
            'fecha_hora': row[2].strftime('%Y-%m-%d %H:%M:%S'),
            'ancho': row[3],
            'alto': row[4],
            'total_pixeles': row[5],
            'matriz_imagen': row[6] if len(row) > 6 else None
        }
        
        objetos = {
            'azul': [],
            'verde': [],
            'rojo': [],
            'amarillo': []
        }
        
        for color in ['azul', 'verde', 'rojo', 'amarillo']:
            cursor.execute(f"SELECT * FROM objetos_{color} WHERE id_foto = ?", foto_id)
            for obj in cursor.fetchall():
                objetos[color].append({
                    'id_objeto': obj[0],
                    'x': obj[2],
                    'y': obj[3],
                    'area': obj[4],
                    'fecha_deteccion': obj[5].strftime('%Y-%m-%d %H:%M:%S') if obj[5] else 'N/A'
                })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'foto': foto,
            'objetos': objetos
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_db_object_details/<color>/<int:obj_id>')
def get_db_object_details(color, obj_id):
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'success': False, 'error': 'No se pudo conectar a la base de datos'})
            
        cursor = conn.cursor()
        
        tabla = f"objetos_{color}"
        cursor.execute(f"SELECT * FROM {tabla} WHERE id_objeto = ?", obj_id)
        row = cursor.fetchone()
        
        if not row:
            return jsonify({'success': False, 'error': 'Objeto no encontrado'})
        
        objeto = {
            'id_objeto': row[0],
            'id_foto': row[1],
            'x': row[2],
            'y': row[3],
            'area': row[4],
            'fecha_deteccion': row[5].strftime('%Y-%m-%d %H:%M:%S') if row[5] else 'N/A'
        }
        
        conn.close()
        
        return jsonify({
            'success': True,
            'objeto': objeto
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/delete_db_object/<color>/<int:obj_id>', methods=['DELETE'])
def delete_db_object(color, obj_id):
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'success': False, 'error': 'No se pudo conectar a la base de datos'})
            
        cursor = conn.cursor()
        
        tabla = f"objetos_{color}"
        cursor.execute(f"DELETE FROM {tabla} WHERE id_objeto = ?", obj_id)
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Objeto eliminado correctamente'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    if not os.path.exists('saved_photos'):
        os.makedirs('saved_photos')
    
    app.run(debug=True, host='0.0.0.0', port=5000)