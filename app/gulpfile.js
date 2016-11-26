'use strict';
 // import
var gulp = require('gulp');
var sass = require('gulp-sass');
var jade = require('gulp-jade');

// jade
var jadeInputDir = './dev/jade/**/*.jade'
var jadeOutputDir = './templates/'
gulp.task('jade', function() {
  var YOUR_LOCALS = {}; // wat
  return gulp.src(jadeInputDir)
    .pipe(jade({locals: YOUR_LOCALS}))
    .pipe(gulp.dest(jadeOutputDir))
});

gulp.task('jade:watch', function () {
  gulp.watch(jadeInputDir, ['jade']);
});

// SASS
var sassInputDir = './dev/sass/**/*.sass'
var sassOutputDir  = './static/css/'
gulp.task('sass', function () {
  gulp.src(sassInputDir)
    .pipe(sass().on('error', sass.logError))
    .pipe(gulp.dest(sassOutputDir));
});

gulp.task('sass:watch', function () {
    gulp.watch(sassInputDir, ['sass']);
});

// Run default
gulp.task('default',[    
    'sass','sass:watch',
    'jade','jade:watch',
]);


