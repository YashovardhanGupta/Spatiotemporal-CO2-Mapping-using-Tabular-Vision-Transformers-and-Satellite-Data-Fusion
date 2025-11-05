// Define the polygon
var chennaiRegion = ee.Geometry.Polygon(
    [[[80.05, 12.75], [80.05, 13.40], [80.50, 13.40], [80.50, 12.75]]], null, false
  );
  
  // Define the time range
  var startDate = ee.Date('2024-01-01');
  var endDate = ee.Date('2024-05-31');
  
  // Generate a list of dates between startDate and endDate
  var dateList = ee.List.sequence(0, endDate.difference(startDate, 'day').subtract(1)).map(function(day) {
    return startDate.advance(day, 'day').format('YYYY-MM-dd');
  });
  
  // Function to create daily composites for NO2
  var createDailyComposite = function(dateStr, regionGeom) {
    var date = ee.Date(dateStr);
    var nextDate = date.advance(1, 'day');
    
    var dailyCollection = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
      .filterDate(date, nextDate)
      .filterBounds(regionGeom)
      .select('NO2_column_number_density');
    
    var dailyImage = dailyCollection.mean()
      .clip(regionGeom)
      .set({
        'date': date.format('YYYY-MM-dd'),
        'system:time_start': date.millis()
      });
  
    return dailyImage;
  };
  
  // Visualization parameters for NO2
  var visParams = {
    min: 0.0,
    max: 0.0002,
    palette: ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']
  };
  
  // Export daily NO2 images
  var visualizeAndExportDailyComposites = function(region) {
    dateList.getInfo().forEach(function(dateStr) {
      var dailyImage = createDailyComposite(dateStr, chennaiRegion).visualize(visParams);
      var dateString = ee.Date(dateStr).format('YYYY-MM-dd').getInfo();
      
      Export.image.toDrive({
        image: dailyImage,
        description: 'chennai_NO2_' + dateString,
        folder: 'NO2_big_01',
        fileNamePrefix: 'chennai_NO2_' + dateString,
        region: chennaiRegion,
        scale: 100,
        crs: 'EPSG:4326',
        maxPixels: 2e10
      });
    });
  };
  
  // Visualize and export the daily composites
  visualizeAndExportDailyComposites(chennaiRegion);
  
  // Center the map for visualization
  Map.centerObject(chennaiRegion, 8);